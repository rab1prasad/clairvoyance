"""N-WID-3 — a paused widget previews only behind the console login.

Before: an ``Origin: https://breezebuddy.ai`` header let any client open a
session on a PAUSED widget and skip the store's allow-list. Any non-browser
client can set that header. Now the public ``/widget/session`` path has no
exception at all, and the console opens its preview session through
``POST /widget-config/{id}/preview/session`` behind RBAC. That session's
widget token carries a server-signed ``preview`` claim and lives an hour.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from app.api.routers.breeze_buddy import widget_common
from app.api.routers.breeze_buddy.widget import handlers as widget_handlers
from app.api.routers.breeze_buddy.widget_config import handlers as config_handlers
from app.api.security.breeze_buddy.widget_token import (
    mint_widget_token,
    require_widget_session,
)
from app.core.middleware.widget_cors_bypass import validated_origin
from app.schemas import UserInfo, UserRole
from app.schemas.breeze_buddy.chat import (
    ChatSessionStatus,
    CreateWidgetSessionRequest,
    WidgetSessionOptions,
    WidgetSurfaceWire,
)
from app.schemas.breeze_buddy.widget_config import WidgetConfigResponse
from app.services.preview.uap.accessor import BOUND_TEMPLATE_VARS, LEDGER_METADATA_KEY

WIDGET = "00000000-0000-0000-0000-0000000000aa"
CONSOLE = "https://breezebuddy.ai"
STORE = "https://store.example"


def _cfg(*, active: bool) -> WidgetConfigResponse:
    return WidgetConfigResponse(
        id=WIDGET,
        reseller_id="buddy-assist",
        merchant_id="store.myshopify.com",
        public_widget_key="public-key-123",
        template_id="00000000-0000-0000-0000-000000000002",
        allowed_origins=[STORE],
        active=active,
    )


def _request(origin: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/agent/voice/breeze-buddy/widget/session",
            "headers": [(b"origin", origin.encode())],
            "client": ("203.0.113.9", 51234),
            "query_string": b"",
        }
    )


def _token_ctx(token: str, request: Request | None = None):
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return require_widget_session(
        request=request or _request(STORE), session_id="s1", creds=creds
    )


def _token_origin(token: str) -> str | None:
    """The origin a request with this token is answered for."""
    request = _request("https://some-other-site.example")
    _token_ctx(token, request)
    return validated_origin(request)


@pytest.fixture()
def by_key(monkeypatch):
    def install(cfg: WidgetConfigResponse) -> None:
        async def lookup(_key: str):
            return cfg

        async def no_limit(**_kwargs) -> None:
            return None

        monkeypatch.setattr(widget_common, "get_widget_config_by_public_key", lookup)
        monkeypatch.setattr(widget_common, "_enforce_widget_ip_limit", no_limit)

    return install


async def _resolve(origin: str):
    return await widget_common.resolve_widget_config_for_request(
        request=_request(origin),
        public_widget_key="public-key-123",
        rate_bucket="session_create",
    )


# -- the public door has no exception --------------------------------------


@pytest.mark.asyncio
async def test_console_origin_no_longer_opens_a_paused_widget(by_key):
    by_key(_cfg(active=False))
    with pytest.raises(HTTPException) as raised:
        await _resolve(CONSOLE)
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_console_origin_no_longer_skips_the_allow_list(by_key):
    by_key(_cfg(active=True))
    with pytest.raises(HTTPException) as raised:
        await _resolve(CONSOLE)
    assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_storefronts_are_unchanged(by_key):
    by_key(_cfg(active=True))
    assert (await _resolve(STORE)).id == WIDGET


def test_the_public_body_cannot_ask_for_a_preview():
    body = CreateWidgetSessionRequest(
        public_widget_key="public-key-123", preview_grant="x", preview=True
    )
    assert "preview_grant" not in body.model_dump()
    assert "preview" not in body.model_dump()


# -- the session both doors open -------------------------------------------


@pytest.fixture()
def opened(monkeypatch):
    """Stub what ``open_widget_session`` reads; capture what it persists."""
    template = SimpleNamespace(secrets={}, merchant_id="store.myshopify.com")
    captured = {}

    async def create(create_req, *_args, **_kwargs):
        captured["metadata"] = create_req.metadata
        captured["template_vars"] = create_req.template_vars
        return SimpleNamespace(
            session_id="s1",
            status=ChatSessionStatus.ACTIVE,
            current_node=None,
            greeting=None,
        )

    monkeypatch.setattr(
        widget_handlers, "get_template_by_id_cached", AsyncMock(return_value=template)
    )
    monkeypatch.setattr(widget_handlers, "validate_template_for_chat", lambda _t: None)
    monkeypatch.setattr(widget_handlers, "negotiate_catalog", lambda _t, _v: ("v1", []))
    monkeypatch.setattr(widget_handlers, "create_chat_session_handler", create)
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
    return captured


def _user(merchant_ids: list[str]) -> UserInfo:
    return UserInfo(
        id="u1",
        username="u1",
        role=UserRole.MERCHANT,
        reseller_ids=["buddy-assist"],
        merchant_ids=merchant_ids,
        permissions=[],
    )


async def _preview(monkeypatch, *, active: bool, merchant_ids: list[str]):
    monkeypatch.setattr(
        config_handlers,
        "get_widget_config_by_id",
        AsyncMock(return_value=_cfg(active=active)),
    )
    return await config_handlers.create_preview_session_handler(
        WIDGET, WidgetSessionOptions(), _user(merchant_ids)
    )


@pytest.mark.asyncio
async def test_the_owner_previews_a_paused_widget(monkeypatch, opened):
    out = await _preview(
        monkeypatch, active=False, merchant_ids=["store.myshopify.com"]
    )
    ctx = _token_ctx(out.widget_token)
    assert (out.session_id, ctx.widget_config_id, ctx.preview) == ("s1", WIDGET, True)
    assert out.ttl_seconds == 60 * 60
    assert opened["metadata"]["widget"]["widget_config_id"] == WIDGET


@pytest.mark.asyncio
async def test_another_merchant_cannot_preview(monkeypatch, opened):
    with pytest.raises(HTTPException) as raised:
        await _preview(monkeypatch, active=False, merchant_ids=["other.myshopify.com"])
    assert raised.value.status_code == 403
    assert opened == {}


@pytest.mark.asyncio
async def test_a_storefront_session_is_not_a_preview(monkeypatch, by_key, opened):
    by_key(_cfg(active=True))
    out = await widget_handlers.create_widget_session_handler(
        CreateWidgetSessionRequest(public_widget_key="public-key-123"),
        _request(STORE),
    )
    assert _token_ctx(out.widget_token).preview is False
    assert out.ttl_seconds == 24 * 60 * 60


@pytest.mark.asyncio
async def test_a_storefront_session_is_bound_to_the_store_origin(
    monkeypatch, by_key, opened
):
    by_key(_cfg(active=True))
    out = await widget_handlers.create_widget_session_handler(
        CreateWidgetSessionRequest(public_widget_key="public-key-123"),
        _request(STORE),
    )
    assert _token_origin(out.widget_token) == STORE


@pytest.mark.asyncio
async def test_a_visitor_cannot_plant_the_uap_binding_or_ledger(
    monkeypatch, by_key, opened
):
    # The payment tools read the rider from template_vars and the ledger
    # from metadata; only the server's UAP binding may write them.
    by_key(_cfg(active=True))
    await widget_handlers.create_widget_session_handler(
        CreateWidgetSessionRequest(
            public_widget_key="public-key-123",
            template_vars={
                "rider_ref": "victims-customer-id",
                "session_id": "victims-session-id",
                "rider_mobile": "9999999999",
                "rider_token": "victims-ny-token",
                "customer_name": "Asha",
            },
            metadata={
                "uap_draws": [{"amount": "-804.00", "status": "drawn"}],
                "utm_source": "ad",
            },
        ),
        _request(STORE),
    )
    for key in BOUND_TEMPLATE_VARS:
        assert key not in opened["template_vars"]
    assert opened["template_vars"]["customer_name"] == "Asha"
    assert LEDGER_METADATA_KEY not in opened["metadata"]
    assert opened["metadata"]["utm_source"] == "ad"


@pytest.mark.asyncio
async def test_a_preview_session_is_bound_to_no_origin(monkeypatch, opened):
    out = await _preview(monkeypatch, active=True, merchant_ids=["store.myshopify.com"])
    assert _token_origin(out.widget_token) is None


@pytest.mark.asyncio
async def test_a_refused_origin_is_never_recorded_as_validated(by_key):
    by_key(_cfg(active=True))
    request = _request(CONSOLE)
    with pytest.raises(HTTPException):
        await widget_common.resolve_widget_config_for_request(
            request=request,
            public_widget_key="public-key-123",
            rate_bucket="session_create",
        )
    assert validated_origin(request) is None


# -- the claim -------------------------------------------------------------


def test_preview_claim_rides_the_session_token():
    plain = mint_widget_token(widget_session_id="s1", widget_config_id=WIDGET)
    preview = mint_widget_token(
        widget_session_id="s1", widget_config_id=WIDGET, preview=True
    )
    assert _token_ctx(plain).preview is False
    assert _token_ctx(preview).preview is True


def test_origin_claim_rides_the_session_token():
    bound = mint_widget_token(
        widget_session_id="s1", widget_config_id=WIDGET, origin=STORE
    )
    unbound = mint_widget_token(widget_session_id="s1", widget_config_id=WIDGET)
    assert _token_origin(bound) == STORE
    assert _token_origin(unbound) is None
