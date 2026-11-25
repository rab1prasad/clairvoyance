"""The generic model provider layer (app/services/model_provider)."""

import json
from typing import Any, Callable, Dict, List

import httpx
import pytest

from app.services.model_provider import (
    CustomSystemOneProvider,
    GenerateRequest,
    GenerationSettings,
    Message,
    OpenRouterProvider,
    ProviderConfig,
    ProviderError,
    TypeSafeProvider,
    Usage,
    close_all,
    resolve_provider,
)

_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number", "minimum": 0, "maximum": 10}},
}


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> OpenRouterProvider:
    provider = OpenRouterProvider(
        ProviderConfig(
            api_key="key", base_url="https://or.test/chat", timeout_seconds=5
        )
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _reply(content: str, finish_reason: str = "stop") -> Dict[str, Any]:
    return {
        "model": "openai/gpt-4o-mini-2024",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def _request(**overrides: Any) -> GenerateRequest:
    fields: Dict[str, Any] = {
        "model": "openai/gpt-4o-mini",
        "input": [Message(role="user", content="hi")],
    }
    fields.update(overrides)
    return GenerateRequest(**fields)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("app.services.model_provider.openrouter.asyncio.sleep", instant)
    monkeypatch.setattr(
        "app.services.model_provider.system_one.typesafe.asyncio.sleep",
        instant,
    )


# --- request ------------------------------------------------------------------


def test_body_plain_text() -> None:
    body = _provider(lambda r: httpx.Response(200)).body(
        _request(
            system_prompt="be brief",
            settings=GenerationSettings(temperature=0, max_tokens=50, stop=["\n"]),
        )
    )
    assert body == {
        "temperature": 0,
        "max_tokens": 50,
        "stop": ["\n"],
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
    }


def test_body_schema_becomes_strict_response_format() -> None:
    body = _provider(lambda r: httpx.Response(200)).body(
        _request(schema=_SCHEMA, extra={"provider": {"order": ["openai"]}})
    )
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }
    assert body["provider"] == {"order": ["openai"], "require_parameters": True}


def test_extra_never_overrides_the_request() -> None:
    body = _provider(lambda r: httpx.Response(200)).body(
        _request(
            settings=GenerationSettings(temperature=0.2),
            extra={"model": "other", "stream": True, "temperature": 1, "seed": 7},
        )
    )
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert body["seed"] == 7


async def test_openrouter_rejects_mapping_input() -> None:
    calls: List[int] = []
    provider = _provider(lambda r: calls.append(1) or httpx.Response(200))
    with pytest.raises(ProviderError, match="sequence of messages") as caught:
        await provider.generate(_request(input={"state": {}}))
    assert not caught.value.retryable
    assert calls == []


# --- generate -----------------------------------------------------------------


async def test_generate_text() -> None:
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer key"
        return httpx.Response(200, json=_reply("hello"))

    response = await _provider(handler).generate(_request())
    assert response.content == "hello"
    assert response.model == "openai/gpt-4o-mini-2024"
    assert response.structured is None
    assert response.finish_reason == "stop"
    assert response.usage == Usage(input_tokens=12, output_tokens=3)
    assert len(seen) == 1


async def test_generate_structured_strips_fences() -> None:
    provider = _provider(
        lambda r: httpx.Response(200, json=_reply('```json\n{"score": 7}\n```'))
    )
    response = await provider.generate(_request(schema=_SCHEMA))
    assert response.structured == {"score": 7}


async def test_structured_reply_not_json_raises() -> None:
    provider = _provider(lambda r: httpx.Response(200, json=_reply("seven")))
    with pytest.raises(ProviderError, match="not JSON"):
        await provider.generate(_request(schema=_SCHEMA))


async def test_structured_truncated_raises() -> None:
    provider = _provider(
        lambda r: httpx.Response(200, json=_reply('{"sco', finish_reason="length"))
    )
    with pytest.raises(ProviderError, match="truncated"):
        await provider.generate(_request(schema=_SCHEMA))


async def test_reply_spent_on_thinking_is_a_truncated_reply() -> None:
    thinking_only = _reply("", finish_reason="length")
    thinking_only["choices"][0]["message"]["content"] = None
    provider = _provider(lambda r: httpx.Response(200, json=thinking_only))
    response = await provider.generate(_request())
    assert (response.content, response.finish_reason) == ("", "length")
    with pytest.raises(ProviderError, match="truncated"):
        await provider.generate(_request(schema=_SCHEMA))


async def test_retries_429_then_succeeds() -> None:
    replies = [
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json=_reply("ok")),
    ]
    provider = _provider(lambda r: replies.pop(0))
    assert (await provider.generate(_request())).content == "ok"
    assert replies == []


async def test_vendor_error_in_200_body_is_retried() -> None:
    replies = [
        httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}),
        httpx.Response(200, json=_reply("ok")),
    ]
    provider = _provider(lambda r: replies.pop(0))
    assert (await provider.generate(_request())).content == "ok"


async def test_vendor_error_code_as_string_or_missing_is_retried() -> None:
    # OpenRouter may send the code as a string, or no code at all
    for error in ({"code": "502", "message": "upstream"}, {"message": "upstream"}):
        replies = [
            httpx.Response(200, json={"error": error}),
            httpx.Response(200, json=_reply("ok")),
        ]
        provider = _provider(lambda r: replies.pop(0))
        assert (await provider.generate(_request())).content == "ok"
        assert replies == []

    # a final code (a string 404 too) is still not retried
    calls: List[int] = []

    def missing(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"error": {"code": "404", "message": "no"}})

    with pytest.raises(ProviderError, match="vendor error 404") as caught:
        await _provider(missing).generate(_request())
    assert calls == [1]
    assert caught.value.status == 404 and not caught.value.retryable


async def test_retries_exhausted_raises() -> None:
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="down")

    with pytest.raises(ProviderError) as caught:
        await _provider(handler).generate(_request())
    assert len(calls) == 3
    assert caught.value.status == 503
    assert caught.value.retryable


_UNROUTABLE = {
    "error": {
        "message": "No endpoints found that can handle the requested parameters.",
        "code": 404,
    }
}


async def test_unroutable_schema_call_fails_loudly_in_one_call() -> None:
    # no endpoint supports every parameter sent: one call, no second try
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json=_UNROUTABLE)

    with pytest.raises(ProviderError, match="requested parameters") as caught:
        await _provider(handler).generate(_request(schema=_SCHEMA))
    assert calls == [1]
    assert caught.value.status == 404 and not caught.value.retryable


async def test_transport_error_is_retried() -> None:
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=_reply("ok"))

    assert (await _provider(handler).generate(_request())).content == "ok"


async def test_client_error_is_not_retried() -> None:
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad model")

    with pytest.raises(ProviderError) as caught:
        await _provider(handler).generate(_request())
    assert len(calls) == 1
    assert caught.value.status == 400
    assert not caught.value.retryable


async def test_no_api_key_raises_without_calling() -> None:
    provider = OpenRouterProvider(
        ProviderConfig(api_key="", base_url="https://or.test", timeout_seconds=5)
    )
    with pytest.raises(ProviderError, match="API key"):
        await provider.generate(_request())
    assert provider._client is None


# --- registry -----------------------------------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider built yet."""
    monkeypatch.setattr("app.services.model_provider._PROVIDERS", {})


@pytest.mark.parametrize(
    "name, kind", [("openrouter", OpenRouterProvider), ("typesafe", TypeSafeProvider)]
)
async def test_a_provider_is_built_once_on_first_use(
    fresh_registry: None, name: str, kind: type
) -> None:
    provider = await resolve_provider({"provider": name})
    assert type(provider) is kind
    # kept, so its pooled client is reused
    assert await resolve_provider({"provider": name}) is provider


async def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(ProviderError, match="unknown provider"):
        await resolve_provider({"provider": "nope"})


async def test_close_all_closes_every_provider_even_if_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: List[str] = []

    class Boom:
        name = "boom"

        async def close(self) -> None:
            raise RuntimeError("pool already gone")

    class Fine:
        name = "fine"

        async def close(self) -> None:
            closed.append("fine")

    monkeypatch.setattr(
        "app.services.model_provider._PROVIDERS",
        {("boom", "", ""): Boom(), ("fine", "", ""): Fine()},
    )
    await close_all()  # must not raise
    assert closed == ["fine"]


async def test_close_all_closes_pools(fresh_registry: None) -> None:
    openrouter = await resolve_provider({"provider": "openrouter"})
    typesafe = await resolve_provider({"provider": "typesafe"})
    assert isinstance(openrouter, OpenRouterProvider)
    assert isinstance(typesafe, TypeSafeProvider)
    openrouter._client = httpx.AsyncClient()
    typesafe._client = httpx.AsyncClient()
    await close_all()
    assert openrouter._client is None
    assert typesafe._client is None


# --- typesafe -----------------------------------------------------------------

_PAYLOAD = {"state": {"transcript": "..."}, "questions": {"polite": {"type": "noul"}}}
_JUDGED = {
    "answers": {"polite": {"noul": 0.9}},
    "model": "jev-2",
    "usage": {"input_tokens": 40, "output_tokens": 6},
}


def _typesafe(handler: Callable[[httpx.Request], httpx.Response]) -> TypeSafeProvider:
    provider = TypeSafeProvider(
        ProviderConfig(api_key="key", base_url="https://ts.test", timeout_seconds=5)
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def test_typesafe_posts_payload_with_model() -> None:
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer key"
        return httpx.Response(200, json=_JUDGED)

    response = await _typesafe(handler).generate(
        GenerateRequest(model="jev", input=_PAYLOAD, system_prompt="ignored")
    )
    assert seen == [{**_PAYLOAD, "model": "jev"}]
    assert response.structured == _JUDGED
    assert json.loads(response.content) == _JUDGED
    assert response.model == "jev-2"
    assert response.usage == Usage(input_tokens=40, output_tokens=6)


async def test_typesafe_rejects_messages_input() -> None:
    calls: List[int] = []
    provider = _typesafe(lambda r: calls.append(1) or httpx.Response(200))
    with pytest.raises(ProviderError, match="mapping"):
        await provider.generate(_request(model="jev"))
    assert calls == []


async def test_typesafe_retries_5xx_then_succeeds() -> None:
    replies = [httpx.Response(504, text="gateway"), httpx.Response(200, json=_JUDGED)]
    provider = _typesafe(lambda r: replies.pop(0))
    response = await provider.generate(GenerateRequest(model="jev", input=_PAYLOAD))
    assert response.structured == _JUDGED
    assert replies == []


async def test_typesafe_client_error_is_not_retried() -> None:
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422, text="bad questions")

    with pytest.raises(ProviderError) as caught:
        await _typesafe(handler).generate(GenerateRequest(model="jev", input=_PAYLOAD))
    assert len(calls) == 1
    assert caught.value.status == 422


# --- custom (System One elsewhere) ---------------------------------------------


@pytest.fixture
def config_keys(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """The config keys the custom provider reads, as a dict; every test starts
    with no endpoint reached yet."""
    values: Dict[str, str] = {}

    async def get_config(key: str, default: Any, return_type: type = str) -> Any:
        return values.get(key, default)

    monkeypatch.setattr("app.services.model_provider.get_config", get_config)
    monkeypatch.setattr("app.services.model_provider._PROVIDERS", {})
    return values


_GRID = {
    "provider": "custom",
    "api_url_name": "GRID_URL",
    "api_key_name": "GRID_KEY",
}


def test_custom_speaks_typesafes_api_under_its_own_name() -> None:
    assert issubclass(CustomSystemOneProvider, TypeSafeProvider)
    assert CustomSystemOneProvider.name == "custom"


@pytest.mark.parametrize(
    "names",
    [
        {"api_url_name": "GRID_URL", "api_key_name": "GRID_KEY"},
        {"api_key_name": "GRID_KEY"},
        {"api_url_name": "GRID_URL"},
    ],
)
async def test_typesafe_always_runs_on_its_own_endpoint(
    config_keys: Dict[str, str], names: Dict[str, str]
) -> None:
    # names are refused when a row is saved; one that got past (an edit in
    # the DB) never sends TypeSafe's key to another host
    config_keys.update(GRID_URL="https://grid.test/v1/systemone", GRID_KEY="k")
    typesafe = await resolve_provider({"provider": "typesafe"})
    assert await resolve_provider({"provider": "typesafe", **names}) is typesafe


async def test_custom_posts_to_the_named_url_with_the_named_key(
    config_keys: Dict[str, str],
) -> None:
    config_keys.update(GRID_URL="https://grid.test/v1/systemone", GRID_KEY="k")
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_JUDGED)

    grid = await resolve_provider(_GRID)
    assert isinstance(grid, CustomSystemOneProvider)
    grid._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = await grid.generate(GenerateRequest(model="jev", input=_PAYLOAD))

    assert str(seen[0].url) == "https://grid.test/v1/systemone"
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert json.loads(seen[0].content) == {**_PAYLOAD, "model": "jev"}
    assert response.structured == _JUDGED
    # one per endpoint, so its pooled client is reused
    assert await resolve_provider(_GRID) is grid


async def test_a_changed_key_is_a_new_endpoint(config_keys: Dict[str, str]) -> None:
    config_keys.update(GRID_URL="https://grid.test", GRID_KEY="old")
    before = await resolve_provider(_GRID)
    config_keys["GRID_KEY"] = "new"
    after = await resolve_provider(_GRID)
    assert isinstance(after, CustomSystemOneProvider) and after is not before
    assert after._config.api_key == "new"


@pytest.mark.parametrize(
    "names, error",
    [
        ({}, "api_url_name and api_key_name are required"),
        ({"api_key_name": "GRID_KEY"}, "api_url_name and api_key_name are required"),
        ({"api_url_name": "GRID_URL"}, "api_url_name and api_key_name are required"),
        ({"api_url_name": "NO_URL", "api_key_name": "GRID_KEY"}, "API URL not found"),
        ({"api_url_name": "GRID_URL", "api_key_name": "NO_KEY"}, "API key not found"),
    ],
)
async def test_custom_without_its_endpoint_is_refused(
    config_keys: Dict[str, str], names: Dict[str, str], error: str
) -> None:
    config_keys.update(GRID_URL="https://grid.test", GRID_KEY="k")
    with pytest.raises(ProviderError, match=error):
        await resolve_provider({"provider": "custom", **names})


async def test_close_all_closes_custom_endpoints(
    config_keys: Dict[str, str],
) -> None:
    config_keys.update(GRID_URL="https://grid.test", GRID_KEY="k")
    grid = await resolve_provider(_GRID)
    assert isinstance(grid, CustomSystemOneProvider)
    grid._client = httpx.AsyncClient()
    await close_all()
    assert grid._client is None
