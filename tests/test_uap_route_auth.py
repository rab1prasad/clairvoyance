"""Who may call the /uap page and chat-tool routes.

Before: every one of them took a ``session_id`` and nothing else, so anyone
who knew a chat session's id could bind a rider to it, read the rider's
payment agents or charge a ticket. Now each needs a login or S2S token whose
scope covers that session's merchant. Juspay's webhook keeps its own token.

The real router and the real login check; the chat session lookup is stubbed
and ``_scope`` — the first thing every route body does — answers 418, so
"418" means "the caller got through the gate and the route ran".
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api.routers.breeze_buddy.preview.uap import handlers
from app.api.security.breeze_buddy.rbac_token import rbac_token_manager
from app.api.security.breeze_buddy.widget_token import mint_widget_token
from app.core.security.jwt import jwt_manager
from app.schemas.breeze_buddy.auth import UserRole
from app.schemas.breeze_buddy.chat import ChatSessionStatus

RESELLER = "namma"
MERCHANT = "namma-yatri"
SESSION = "session-of-namma-yatri"
OTHER_SESSION = "session-of-another-merchant"
ROUTE_RAN = 418

SESSIONS = {
    SESSION: SimpleNamespace(
        id=SESSION,
        reseller_id=RESELLER,
        merchant_id=MERCHANT,
        status=ChatSessionStatus.ACTIVE,
        metadata={"template_vars": {"rider_ref": "customer-1"}},
    ),
    OTHER_SESSION: SimpleNamespace(
        id=OTHER_SESSION,
        reseller_id="other-reseller",
        merchant_id="other-merchant",
        status=ChatSessionStatus.ACTIVE,
        metadata={"template_vars": {"rider_ref": "customer-2"}},
    ),
}


def _calls(session_id: str) -> list[tuple[str, str, dict]]:
    """One valid request per gated route: (method, path, json-or-params)."""
    return [
        (
            "POST",
            "/uap/rider",
            {"session_id": session_id, "object_reference_id": "rider-12345"},
        ),
        ("POST", "/uap/onboarding", {"session_id": session_id}),
        ("GET", "/uap/status", {"session_id": session_id}),
        ("GET", "/uap/agents", {"session_id": session_id}),
        (
            "POST",
            "/uap/agents/select",
            {"session_id": session_id, "agent_ref": "agent-1"},
        ),
        ("POST", "/uap/result", {"session_id": session_id, "agent_obj_ref": "ref-1"}),
        (
            "POST",
            "/uap/draw",
            {
                "session_id": session_id,
                "journey_id": "journey-1",
                "rider_token": "ny-token",
                "amount": "24.00",
            },
        ),
        (
            "GET",
            "/uap/ticket",
            {
                "session_id": session_id,
                "journey_id": "journey-1",
                "rider_token": "ny-token",
            },
        ),
    ]


ROUTES = [f"{method} {path}" for method, path, _ in _calls(SESSION)]


def _login(
    role: UserRole, reseller_ids: list[str], merchant_ids: list[str], **kwargs
) -> str:
    return rbac_token_manager.create_access_token_with_rbac(
        user_id="u1",
        username="tester",
        role=role,
        reseller_ids=reseller_ids,
        merchant_ids=merchant_ids,
        **kwargs,
    )


MERCHANT_LOGIN = _login(UserRole.MERCHANT, [RESELLER], [MERCHANT])
RESELLER_LOGIN = _login(UserRole.RESELLER, [RESELLER], ["*"])
ADMIN_LOGIN = _login(UserRole.ADMIN, ["*"], ["*"])
SIBLING_MERCHANT_LOGIN = _login(UserRole.MERCHANT, [RESELLER], ["another-shop"])
OTHER_RESELLER_LOGIN = _login(UserRole.RESELLER, ["other-reseller"], ["*"])
EXPIRED_LOGIN = _login(
    UserRole.MERCHANT, [RESELLER], [MERCHANT], expires_delta=timedelta(minutes=-5)
)
# The session's own anonymous credentials: signed by us, bound to this very
# session, and still not a merchant's word.
WIDGET_TOKEN = mint_widget_token(
    widget_session_id=SESSION, widget_config_id="00000000-0000-0000-0000-0000000000aa"
)
DEMO_TOKEN = jwt_manager.create_access_token(
    {"sub": "demo-1", "demo": True, "demo_session_id": SESSION},
    expires_delta=timedelta(minutes=30),
)


@pytest.fixture()
def route_body(monkeypatch) -> AsyncMock:
    """The mock every gated route body reaches first."""

    async def lookup(session_id: str):
        return SESSIONS.get(session_id)

    reached = AsyncMock(side_effect=HTTPException(status_code=ROUTE_RAN))
    monkeypatch.setattr(handlers, "get_chat_session_by_id", lookup)
    monkeypatch.setattr(handlers, "_scope", reached)
    return reached


@pytest.fixture()
def client(route_body) -> TestClient:
    app = FastAPI()
    app.include_router(handlers.router, prefix="/uap")
    return TestClient(app)


def _send(client: TestClient, call: tuple[str, str, dict], token: str | None):
    method, path, data = call
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if method == "GET":
        return client.get(path, params=data, headers=headers)
    return client.post(path, json=data, headers=headers)


# -- refused before the route runs -----------------------------------------


@pytest.mark.parametrize("call", _calls(SESSION), ids=ROUTES)
def test_no_login_is_refused(client: TestClient, route_body: AsyncMock, call) -> None:
    resp = _send(client, call, None)
    assert resp.status_code == 401
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls(SESSION), ids=ROUTES)
@pytest.mark.parametrize(
    "token",
    ["not-a-token", EXPIRED_LOGIN, WIDGET_TOKEN, DEMO_TOKEN],
    ids=["garbage", "expired login", "the session's widget token", "a demo token"],
)
def test_a_token_that_is_not_a_valid_login_is_refused(
    client: TestClient, route_body: AsyncMock, call, token: str
) -> None:
    resp = _send(client, call, token)
    assert resp.status_code == 401
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls(SESSION), ids=ROUTES)
@pytest.mark.parametrize(
    "token",
    [SIBLING_MERCHANT_LOGIN, OTHER_RESELLER_LOGIN],
    ids=["another merchant of the reseller", "another reseller"],
)
def test_a_login_for_another_tenant_is_refused(
    client: TestClient, route_body: AsyncMock, call, token: str
) -> None:
    resp = _send(client, call, token)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat session not found"
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls("no-such-session-id"), ids=ROUTES)
def test_an_unknown_session_reads_like_one_out_of_scope(
    client: TestClient, route_body: AsyncMock, call
) -> None:
    resp = _send(client, call, MERCHANT_LOGIN)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat session not found"
    route_body.assert_not_awaited()


@pytest.mark.parametrize(
    "method, path", [("POST", "/uap/onboarding"), ("GET", "/uap/status")]
)
def test_a_request_naming_no_session_is_refused(
    client: TestClient, route_body: AsyncMock, method: str, path: str
) -> None:
    resp = _send(client, (method, path, {}), MERCHANT_LOGIN)
    assert resp.status_code == 400
    route_body.assert_not_awaited()


def test_a_post_is_judged_by_the_session_in_its_body(
    client: TestClient, route_body: AsyncMock
) -> None:
    """The route acts on the body's session, so that is the one checked: a
    session the caller owns in the URL must not vouch for another's in the
    body."""
    resp = client.post(
        "/uap/draw",
        params={"session_id": SESSION},
        json={
            "session_id": OTHER_SESSION,
            "journey_id": "journey-1",
            "rider_token": "ny-token",
            "amount": "24.00",
        },
        headers={"Authorization": f"Bearer {MERCHANT_LOGIN}"},
    )
    assert resp.status_code == 404
    route_body.assert_not_awaited()


# -- let through ------------------------------------------------------------


@pytest.mark.parametrize("call", _calls(SESSION), ids=ROUTES)
@pytest.mark.parametrize(
    "token",
    [MERCHANT_LOGIN, RESELLER_LOGIN, ADMIN_LOGIN],
    ids=["the merchant", "its reseller", "an admin"],
)
def test_a_login_covering_the_sessions_merchant_reaches_the_route(
    client: TestClient, route_body: AsyncMock, call, token: str
) -> None:
    resp = _send(client, call, token)
    assert resp.status_code == ROUTE_RAN
    route_body.assert_awaited_once()


# -- the webhook is Juspay's, not a logged-in caller's ----------------------


def test_the_webhook_needs_its_own_token_not_a_login(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        handlers, "_token_tenant", AsyncMock(return_value=(RESELLER, MERCHANT))
    )

    resp = client.post(
        "/uap/webhook", params={"token": "juspay-token"}, content="not json"
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}


def test_the_webhook_still_refuses_a_caller_without_its_token(
    client: TestClient,
) -> None:
    resp = client.post(
        "/uap/webhook",
        json={},
        headers={"Authorization": f"Bearer {ADMIN_LOGIN}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "unauthorized"


# -- no route is added without the gate ------------------------------------


def test_every_route_but_the_webhook_declares_the_gate() -> None:
    ungated = [
        route.path
        for route in handlers.router.routes
        if isinstance(route, APIRoute)
        and not any(
            dep.call is handlers.require_session_caller
            for dep in route.dependant.dependencies
        )
    ]
    assert ungated == ["/webhook"]
