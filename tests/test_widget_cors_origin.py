"""Which origin the real widget routes name in ``Access-Control-Allow-Origin``.

The real widget router behind the real CORS middleware, mounted where
production mounts it. The DB, the rate limiter and what a session needs to
open are stubbed; the origin allowlist, the widget token and the middleware
run for real — they ARE the subject.

Before: every answer on a widget path said ``*``, so a page on any site that
got hold of a widget token could read the conversation it belongs to. Now the
answer names the origin the session was opened from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.ai.voice.agents.breeze_buddy.assist.platforms.shopify.tenancy import (
    assist_tenant,
)
from app.api.routers.breeze_buddy import widget as widget_routes, widget_common
from app.api.routers.breeze_buddy.widget import handlers as widget_handlers, storefront
from app.api.security.breeze_buddy.widget_token import (
    mint_widget_token,
    require_widget_session,
)
from app.core.middleware.widget_cors_bypass import CustomWidgetCorsBypassMiddleware
from app.schemas.breeze_buddy.chat import (
    ChatEndedReason,
    ChatSessionStatus,
    EndChatSessionResponse,
    WidgetSurfaceWire,
)
from app.schemas.breeze_buddy.widget_config import WidgetConfigResponse
from app.services.redis.rate_limit import RateLimitDecision

PREFIX = "/agent/voice/breeze-buddy"
SHOP = "store.myshopify.com"
STORE = "https://store.example"
OTHER = "https://other.example"
CONSOLE = "https://portal.breeze.in"
WIDGET_ID = "00000000-0000-0000-0000-0000000000aa"
SESSION = "s1"


def _cfg() -> WidgetConfigResponse:
    reseller_id, merchant_id = assist_tenant("buddy-assist", SHOP)
    return WidgetConfigResponse(
        id=WIDGET_ID,
        reseller_id=reseller_id,
        merchant_id=merchant_id,
        public_widget_key="public-key-123",
        template_id="00000000-0000-0000-0000-000000000002",
        allowed_origins=[STORE],
        active=True,
        appearance={"header_title": "Store Assist"},
        updated_at=datetime(2026, 9, 3, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    async def no_limit(**_kwargs) -> None:
        return None

    async def by_key(key: str):
        return _cfg() if key == "public-key-123" else None

    async def by_shop(_reseller_id: str, merchant_id: str):
        return _cfg() if merchant_id == _cfg().merchant_id else None

    # the public door
    monkeypatch.setattr(widget_common, "get_widget_config_by_public_key", by_key)
    monkeypatch.setattr(widget_common, "_enforce_widget_ip_limit", no_limit)
    # the storefront loader's door
    monkeypatch.setattr(storefront, "get_widget_config_by_reseller_merchant", by_shop)
    monkeypatch.setattr(storefront, "enforce_widget_ip_limit", AsyncMock())
    monkeypatch.setattr(
        storefront,
        "check_rate_limit",
        AsyncMock(
            return_value=RateLimitDecision(
                allowed=True, count=1, limit=600, retry_after_seconds=0
            )
        ),
    )
    # what opening a session reads
    template = SimpleNamespace(secrets={}, merchant_id=SHOP)
    monkeypatch.setattr(
        widget_handlers, "get_template_by_id_cached", AsyncMock(return_value=template)
    )
    monkeypatch.setattr(widget_handlers, "validate_template_for_chat", lambda _t: None)
    monkeypatch.setattr(widget_handlers, "negotiate_catalog", lambda _t, _v: ("v1", []))
    monkeypatch.setattr(
        widget_handlers,
        "create_chat_session_handler",
        AsyncMock(
            return_value=SimpleNamespace(
                session_id=SESSION,
                status=ChatSessionStatus.ACTIVE,
                current_node=None,
                greeting=None,
            )
        ),
    )
    monkeypatch.setattr(
        widget_handlers,
        "_extract_widget_config",
        lambda _t: SimpleNamespace(
            quick_replies=[], greeting_tiles=[], enable_text_input=True
        ),
    )
    monkeypatch.setattr(widget_handlers, "_template_voice_enabled", lambda _t: False)
    monkeypatch.setattr(
        widget_handlers, "_surface_wire", AsyncMock(return_value=WidgetSurfaceWire())
    )
    # one follow-up route's work: its answer stands for the conversation
    monkeypatch.setattr(
        widget_routes,
        "end_widget_session_handler",
        AsyncMock(
            return_value=EndChatSessionResponse(
                session_id=SESSION,
                status=ChatSessionStatus.ENDED,
                ended_reason=ChatEndedReason.USER_ENDED,
            )
        ),
    )

    app = FastAPI()
    app.include_router(widget_routes.router, prefix=PREFIX)
    app.add_middleware(
        CustomWidgetCorsBypassMiddleware,
        allow_origins=[CONSOLE],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return TestClient(app)


def _allow_origin(resp) -> list[str]:
    return resp.headers.get_list("access-control-allow-origin")


def _open(client: TestClient, origin: str):
    return client.post(
        f"{PREFIX}/widget/session",
        json={"public_widget_key": "public-key-123"},
        headers={"Origin": origin},
    )


def _end(client: TestClient, token: str, origin: str, session: str = SESSION):
    return client.post(
        f"{PREFIX}/widget/session/{session}/end",
        headers={"Origin": origin, "Authorization": f"Bearer {token}"},
    )


# -- opening a session -----------------------------------------------------


def test_opening_a_session_names_the_store(client: TestClient) -> None:
    resp = _open(client, STORE)
    assert resp.status_code == 201
    assert _allow_origin(resp) == [STORE]


def test_a_site_not_on_the_allowlist_can_read_its_refusal(client: TestClient) -> None:
    resp = _open(client, OTHER)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Origin not permitted for this widget"
    assert _allow_origin(resp) == [OTHER]


def test_an_unknown_widget_key_is_readable_too(client: TestClient) -> None:
    resp = client.post(
        f"{PREFIX}/widget/session",
        json={"public_widget_key": "no-such-key"},
        headers={"Origin": STORE},
    )
    assert resp.status_code == 404
    assert _allow_origin(resp) == [STORE]


# -- the calls that carry the widget token ----------------------------------


def test_the_store_reads_its_own_conversation(client: TestClient) -> None:
    token = _open(client, STORE).json()["widget_token"]

    resp = _end(client, token, STORE)

    assert resp.status_code == 200
    assert _allow_origin(resp) == [STORE]


def test_another_site_with_the_token_cannot_read_the_answer(
    client: TestClient,
) -> None:
    """The request is not refused (a non-browser client could send it anyway);
    the answer names the store, so the other page's browser withholds it."""
    token = _open(client, STORE).json()["widget_token"]

    resp = _end(client, token, OTHER)

    assert resp.status_code == 200
    assert _allow_origin(resp) == [STORE]


def test_a_token_for_another_session_still_names_the_store(
    client: TestClient,
) -> None:
    token = _open(client, STORE).json()["widget_token"]

    resp = _end(client, token, OTHER, session="some-other-session")

    assert resp.status_code == 401
    assert _allow_origin(resp) == [STORE]


def test_an_expired_token_is_a_401_the_embed_can_read(client: TestClient) -> None:
    """The embed opens a fresh session on this 401, so it has to see it."""
    expired = mint_widget_token(
        widget_session_id=SESSION,
        widget_config_id=WIDGET_ID,
        ttl_minutes=-5,
        origin=STORE,
    )

    resp = _end(client, expired, STORE)

    assert resp.status_code == 401
    assert _allow_origin(resp) == [STORE]


def test_a_garbage_token_is_a_401_the_caller_can_read(client: TestClient) -> None:
    resp = _end(client, "not-a-token", OTHER)
    assert resp.status_code == 401
    assert _allow_origin(resp) == [OTHER]


def test_a_token_without_an_origin_names_the_caller(client: TestClient) -> None:
    """A console preview token, and one minted before the claim existed:
    they keep working from the origin they are used on."""
    unbound = mint_widget_token(widget_session_id=SESSION, widget_config_id=WIDGET_ID)

    resp = _end(client, unbound, CONSOLE)

    assert resp.status_code == 200
    assert _allow_origin(resp) == [CONSOLE]


def test_a_followup_preflight_passes_for_any_origin(client: TestClient) -> None:
    """It carries no token, so no session is known yet."""
    resp = client.options(
        f"{PREFIX}/widget/session/{SESSION}/end",
        headers={
            "Origin": OTHER,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert resp.status_code == 204
    assert _allow_origin(resp) == [OTHER]


# -- the storefront loader -------------------------------------------------


def _storefront(client: TestClient, origin: str, shop: str = SHOP, **headers: str):
    return client.get(
        f"{PREFIX}/widget/storefront-config",
        params={"merchant_domain": shop},
        headers={"Origin": origin, **headers},
    )


def test_the_storefront_config_names_the_store(client: TestClient) -> None:
    resp = _storefront(client, STORE)
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert _allow_origin(resp) == [STORE]
    assert resp.headers["Access-Control-Expose-Headers"] == "ETag"


def test_an_unchanged_config_is_a_304_the_loader_can_read(client: TestClient) -> None:
    etag = _storefront(client, STORE).headers["ETag"]

    resp = _storefront(client, STORE, **{"If-None-Match": etag})

    assert resp.status_code == 304
    assert _allow_origin(resp) == [STORE]


def test_the_loader_reads_its_refusal_from_a_site_not_allowed(
    client: TestClient,
) -> None:
    """403 tells the loader to drop its cached copy and unmount."""
    resp = _storefront(client, OTHER)
    assert resp.status_code == 403
    assert _allow_origin(resp) == [OTHER]


def test_the_loader_reads_no_widget_here_from_any_shop(client: TestClient) -> None:
    """A shop can switch the embed on before it is onboarded; the loader has
    to read ``enabled: false`` from an origin no widget knows."""
    resp = _storefront(client, OTHER, shop="not-onboarded.myshopify.com")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    assert _allow_origin(resp) == [OTHER]


# -- no route answers without one of the two checks -------------------------


def test_every_widget_route_validates_an_origin_or_needs_the_token() -> None:
    """An answer names the caller's own origin only when nothing was
    validated. That is safe because every route that returns something of a
    merchant's either checks the origin itself (the two public doors) or
    depends on the widget token. A new route that does neither would hand its
    answer to any site — so it fails here first."""
    public_doors = {"/widget/storefront-config", "/widget/session"}
    unguarded = []
    for route in widget_routes.router.routes:
        if not isinstance(route, APIRoute) or route.methods == {"OPTIONS"}:
            continue
        needs_token = any(
            dep.call is require_widget_session for dep in route.dependant.dependencies
        )
        if not needs_token and route.path not in public_doors:
            unguarded.append(route.path)
    assert unguarded == []
