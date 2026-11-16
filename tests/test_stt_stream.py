"""Tests for the realtime STT stream: schema, WS auth shim, protocol edges."""

import asyncio
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from app.api.routers.breeze_buddy.stt import handlers
from app.api.security.breeze_buddy import rbac_token
from app.schemas import UserInfo
from app.schemas.breeze_buddy.auth import UserRole
from app.schemas.breeze_buddy.stt import TranscriptionStreamRequest

_USER = UserInfo(id="user-1", username="tester", role=UserRole.ADMIN)


class FakeWebSocket:
    """Minimal stand-in implementing the surface the stream handler uses.

    ``first_texts`` are the text messages sent before the config message (the
    auth message of a client that set no header). ``messages`` scripts what
    ``receive()`` yields after the config message; once exhausted, a
    disconnect message is returned. ``fail_sends_after`` makes ``send_text``
    raise once that many messages have been sent.
    """

    def __init__(
        self,
        config_text: str,
        headers: dict | None = None,
        query: dict | None = None,
        messages: list[dict] | None = None,
        fail_sends_after: int | None = None,
        first_texts: list[str] | None = None,
    ):
        self._first_texts = list(first_texts or [])
        self._config_text = config_text
        self._messages = list(messages or [])
        self._fail_sends_after = fail_sends_after
        self.headers = headers or {}
        self.query_params = query or {}
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None
        self.accepted = False
        self.client_state = SimpleNamespace(name="CONNECTED")

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self._first_texts:
            return self._first_texts.pop(0)
        return self._config_text

    async def receive(self) -> dict:
        if self._messages:
            return self._messages.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_text(self, text: str) -> None:
        if (
            self._fail_sends_after is not None
            and len(self.sent) >= self._fail_sends_after
        ):
            raise RuntimeError("send failed")
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)
        self.client_state = SimpleNamespace(name="DISCONNECTED")


def as_ws(fake: FakeWebSocket) -> WebSocket:
    return cast(WebSocket, fake)


class StubTranscriber:
    """Replaces StreamingTranscriber so handler tests skip Pipecat entirely."""

    def __init__(self, stt_service, *, sample_rate, on_transcript):
        self.on_transcript = on_transcript
        self.fed: list[bytes] = []
        self.stopped = False
        self.failed = False
        self.start_error: str | None = None
        self.emit_on_feed = False

    async def start(self) -> None:
        if self.start_error:
            raise RuntimeError(self.start_error)

    async def feed(self, audio: bytes) -> None:
        self.fed.append(audio)
        if self.emit_on_feed:
            from app.ai.voice.stt.streaming import TranscriptEvent

            await self.on_transcript(TranscriptEvent(text="hi", is_final=True))

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def stream_env(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch auth, provider factory, limits, and the transcriber."""
    created: dict = {}

    def _make(stt_service, *, sample_rate, on_transcript):
        stub = StubTranscriber(
            stt_service, sample_rate=sample_rate, on_transcript=on_transcript
        )
        created["transcriber"] = stub
        for attr, value in created.get("presets", {}).items():
            setattr(stub, attr, value)
        return stub

    monkeypatch.setattr(
        handlers, "get_user_from_websocket", AsyncMock(return_value=_USER)
    )
    create_mock = AsyncMock(return_value=object())
    monkeypatch.setattr(handlers, "create_stt_from_config", create_mock)
    created["create_stt"] = create_mock
    monkeypatch.setattr(handlers, "STT_STREAM_MAX_SECONDS", AsyncMock(return_value=300))
    monkeypatch.setattr(
        handlers, "STT_STREAM_IDLE_TIMEOUT_SECONDS", AsyncMock(return_value=60)
    )
    monkeypatch.setattr(handlers, "StreamingTranscriber", _make)
    return created


def test_stream_request_normalizes_and_bounds() -> None:
    request = TranscriptionStreamRequest.model_validate(
        {"provider": " SONIOX ", "model": "  ", "language": ["en", " ", "hi "]}
    )
    assert request.provider.value == "soniox"
    assert request.model is None
    assert request.language == ["en", "hi"]
    assert request.sample_rate == 16000

    with pytest.raises(ValueError):
        TranscriptionStreamRequest.model_validate(
            {"provider": "soniox", "sample_rate": 4000}
        )


_AUTH_MESSAGE = json.dumps({"type": "auth", "token": "message-token"})


@pytest.fixture
def verified_tokens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every token is accepted, and the ones that reached the check are listed."""
    seen: list[str] = []
    monkeypatch.setattr(
        rbac_token.rbac_token_manager,
        "verify_rbac_token",
        lambda token: seen.append(token) or _USER,
    )
    return seen


async def test_websocket_auth_reads_header_then_first_message(
    verified_tokens: list[str],
) -> None:
    ws = FakeWebSocket("", headers={"authorization": "Bearer header-token"})
    assert await rbac_token.get_user_from_websocket(as_ws(ws)) is _USER

    ws = FakeWebSocket("", first_texts=[_AUTH_MESSAGE])
    assert await rbac_token.get_user_from_websocket(as_ws(ws)) is _USER
    assert verified_tokens == ["header-token", "message-token"]


async def test_websocket_auth_with_a_header_leaves_the_first_message_unread(
    verified_tokens: list[str],
) -> None:
    ws = FakeWebSocket(
        "", headers={"authorization": "Bearer header-token"}, first_texts=["config"]
    )
    await rbac_token.get_user_from_websocket(as_ws(ws))

    assert await ws.receive_text() == "config"


async def test_websocket_auth_does_not_read_a_token_from_the_url(
    verified_tokens: list[str],
) -> None:
    ws = FakeWebSocket("{}", query={"token": "query-token"})

    with pytest.raises(HTTPException) as refused:
        await rbac_token.get_user_from_websocket(as_ws(ws))

    assert refused.value.status_code == 401
    assert verified_tokens == []


@pytest.mark.parametrize(
    "first_message",
    [
        '{"provider": "soniox"}',
        '{"type": "auth"}',
        '{"type": "auth", "token": 7}',
        '{"type": "auth", "token": "  "}',
        '{"type": "stop", "token": "message-token"}',
        '["auth", "message-token"]',
        "not json",
        # json.loads raises RecursionError for this, not a ValueError.
        pytest.param("[" * 10_000, id="nested too deep to parse"),
    ],
)
async def test_websocket_auth_refuses_a_first_message_that_is_not_auth(
    verified_tokens: list[str], first_message: str
) -> None:
    ws = FakeWebSocket("", first_texts=[first_message])

    with pytest.raises(HTTPException) as refused:
        await rbac_token.get_user_from_websocket(as_ws(ws))

    assert refused.value.status_code == 401
    assert verified_tokens == []


@pytest.mark.parametrize(
    "failure",
    [WebSocketDisconnect(), KeyError("text")],
    ids=["client disconnects", "binary frame"],
)
async def test_websocket_auth_refuses_when_no_text_message_arrives(
    monkeypatch: pytest.MonkeyPatch, verified_tokens: list[str], failure: Exception
) -> None:
    ws = FakeWebSocket("")
    monkeypatch.setattr(ws, "receive_text", AsyncMock(side_effect=failure))

    with pytest.raises(HTTPException) as refused:
        await rbac_token.get_user_from_websocket(as_ws(ws))

    assert refused.value.status_code == 401
    assert verified_tokens == []


async def test_websocket_auth_stops_waiting_for_the_auth_message(
    monkeypatch: pytest.MonkeyPatch, verified_tokens: list[str]
) -> None:
    async def _never_sends() -> str:
        await asyncio.sleep(60)
        return _AUTH_MESSAGE

    ws = FakeWebSocket("")
    monkeypatch.setattr(ws, "receive_text", _never_sends)
    monkeypatch.setattr(rbac_token, "WEBSOCKET_AUTH_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(HTTPException) as refused:
        await asyncio.wait_for(rbac_token.get_user_from_websocket(as_ws(ws)), timeout=5)

    assert refused.value.status_code == 401
    assert verified_tokens == []


async def test_stream_rejects_unauthenticated() -> None:
    ws = FakeWebSocket("{}")
    await handlers.handle_transcription_stream(as_ws(ws))

    assert ws.accepted
    assert ws.sent[-1]["type"] == "error"
    assert ws.closed is not None and ws.closed[0] == 4401


async def test_stream_refuses_a_first_message_nested_too_deep_to_parse() -> None:
    """An anonymous caller must get the same 4401 as any other bad first
    message, not an unhandled error (close 1011 and a traceback)."""
    ws = FakeWebSocket("{}", first_texts=["[" * 10_000])
    await handlers.handle_transcription_stream(as_ws(ws))

    assert ws.sent[-1] == {"type": "error", "message": "Missing credentials"}
    assert ws.closed is not None and ws.closed[0] == 4401


async def test_stream_rejects_a_token_in_the_url(
    stream_env: dict, monkeypatch: pytest.MonkeyPatch, verified_tokens: list[str]
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", rbac_token.get_user_from_websocket
    )

    ws = FakeWebSocket(
        json.dumps({"provider": "soniox"}), query={"token": "query-token"}
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert verified_tokens == []
    assert all(event["type"] != "ready" for event in ws.sent)
    assert ws.closed is not None and ws.closed[0] == 4401
    stream_env["create_stt"].assert_not_awaited()


async def test_stream_takes_the_token_from_the_first_message_then_the_config(
    stream_env: dict, monkeypatch: pytest.MonkeyPatch, verified_tokens: list[str]
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", rbac_token.get_user_from_websocket
    )

    ws = FakeWebSocket(
        json.dumps({"provider": "deepgram"}), first_texts=[_AUTH_MESSAGE]
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert verified_tokens == ["message-token"]
    assert ws.sent[0]["type"] == "ready"
    assert stream_env["create_stt"].await_args.args[0].provider.value == "deepgram"


async def test_stream_with_a_header_token_reads_the_config_first(
    stream_env: dict, monkeypatch: pytest.MonkeyPatch, verified_tokens: list[str]
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", rbac_token.get_user_from_websocket
    )

    ws = FakeWebSocket(
        json.dumps({"provider": "deepgram"}),
        headers={"authorization": "Bearer header-token"},
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert verified_tokens == ["header-token"]
    assert ws.sent[0]["type"] == "ready"


async def test_stream_rejects_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", AsyncMock(return_value=_USER)
    )

    ws = FakeWebSocket(json.dumps({"provider": "not-a-provider"}))
    await handlers.handle_transcription_stream(as_ws(ws))

    assert ws.sent[-1]["type"] == "error"
    assert ws.closed is not None and ws.closed[0] == 4400


async def test_stream_rejects_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", AsyncMock(return_value=_USER)
    )

    ws = FakeWebSocket(json.dumps({"provider": "openai"}))
    await handlers.handle_transcription_stream(as_ws(ws))

    assert ws.sent[-1]["type"] == "error"
    assert "realtime" in ws.sent[-1]["message"]
    assert ws.closed is not None and ws.closed[0] == 4400


async def test_stream_rejects_sarvam_sample_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        handlers, "get_user_from_websocket", AsyncMock(return_value=_USER)
    )

    ws = FakeWebSocket(json.dumps({"provider": "sarvam", "sample_rate": 8000}))
    await handlers.handle_transcription_stream(as_ws(ws))

    assert ws.sent[-1]["type"] == "error"
    assert "sample_rate" in ws.sent[-1]["message"]
    assert ws.closed is not None and ws.closed[0] == 4400


async def test_stream_rejects_provider_startup_failure(stream_env: dict) -> None:
    stream_env["presets"] = {"start_error": "connect refused"}

    ws = FakeWebSocket(json.dumps({"provider": "soniox"}))
    await handlers.handle_transcription_stream(as_ws(ws))

    assert all(event["type"] != "ready" for event in ws.sent)
    assert ws.sent[-1]["type"] == "error"
    assert "provider connection failed" in ws.sent[-1]["message"]
    assert ws.closed is not None and ws.closed[0] == 4400


async def test_stream_rejects_oversized_chunk(stream_env: dict) -> None:
    big = b"x" * (handlers._MAX_CHUNK_BYTES + 1)
    ws = FakeWebSocket(
        json.dumps({"provider": "soniox"}),
        messages=[{"type": "websocket.receive", "bytes": big}],
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert stream_env["transcriber"].fed == []
    assert stream_env["transcriber"].stopped
    assert ws.sent[-1]["type"] == "error"
    assert "chunk exceeds" in ws.sent[-1]["message"]
    assert ws.closed is not None and ws.closed[0] == 4400


async def test_stream_feeds_audio_and_forwards_transcripts(stream_env: dict) -> None:
    stream_env["presets"] = {"emit_on_feed": True}
    chunk = b"\x00\x01" * 100
    ws = FakeWebSocket(
        json.dumps({"provider": "soniox"}),
        messages=[
            {"type": "websocket.receive", "bytes": chunk},
            {"type": "websocket.receive", "text": json.dumps({"type": "stop"})},
        ],
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert stream_env["transcriber"].fed == [chunk]
    assert stream_env["transcriber"].stopped
    types = [event["type"] for event in ws.sent]
    assert types[0] == "ready"
    assert "final" in types
    assert ws.closed is not None and ws.closed[0] == 1000


async def test_stream_passes_nested_provider_config(stream_env: dict) -> None:
    ws = FakeWebSocket(
        json.dumps(
            {
                "provider": "soniox",
                "soniox": {"context": "acme brands", "model": "stt-rt-v4"},
            }
        )
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    config = stream_env["create_stt"].await_args.args[0]
    assert config.provider.value == "soniox"
    assert config.soniox is not None
    assert config.soniox.context == "acme brands"
    assert config.soniox.model == "stt-rt-v4"


async def test_stream_flat_model_builds_nested_config(stream_env: dict) -> None:
    ws = FakeWebSocket(json.dumps({"provider": "deepgram", "model": "nova-3-medical"}))
    await handlers.handle_transcription_stream(as_ws(ws))

    config = stream_env["create_stt"].await_args.args[0]
    assert config.deepgram is not None
    assert config.deepgram.model == "nova-3-medical"
    assert config.soniox is None


async def test_stream_flat_model_fills_nested_config_without_model(
    stream_env: dict,
) -> None:
    ws = FakeWebSocket(
        json.dumps(
            {
                "provider": "deepgram",
                "model": "nova-3-medical",
                "deepgram": {"endpointing_ms": 50},
            }
        )
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    config = stream_env["create_stt"].await_args.args[0]
    assert config.deepgram is not None
    assert config.deepgram.model == "nova-3-medical"
    assert config.deepgram.endpointing_ms == 50


async def test_stream_ends_when_client_send_fails(stream_env: dict) -> None:
    stream_env["presets"] = {"emit_on_feed": True}
    chunk = b"\x00\x01" * 100
    # Allow only the "ready" send; the transcript forward then fails.
    ws = FakeWebSocket(
        json.dumps({"provider": "soniox"}),
        messages=[
            {"type": "websocket.receive", "bytes": chunk},
            {"type": "websocket.receive", "bytes": chunk},
        ],
        fail_sends_after=1,
    )
    await handlers.handle_transcription_stream(as_ws(ws))

    assert stream_env["transcriber"].stopped
    assert ws.closed is not None and ws.closed[0] == 1011
    assert ws.closed[1] == "client send failed"
