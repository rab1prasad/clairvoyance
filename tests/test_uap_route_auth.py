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

import app.crm.auth as crm_auth
from app.api.routers.breeze_buddy.preview.uap import handlers
from app.api.security.breeze_buddy.rbac_token import rbac_token_manager
from app.api.security.breeze_buddy.widget_token import mint_widget_token
from app.core.security.jwt import jwt_manager
from app.schemas.breeze_buddy.auth import UserRole
from app.schemas.breeze_buddy.chat import ChatSessionStatus

RESELLER = "namma"
MERCHANT = "namma-yatri"
# Chat session ids are UUIDs; the gate refuses anything else before the lookup.
SESSION = "11111111-1111-4111-8111-111111111111"  # namma-yatri's
OTHER_SESSION = "22222222-2222-4222-8222-222222222222"  # another merchant's
# Labelled with the caller's reseller, but its merchant is another reseller's
# (a template can name any merchant).
LABELLED_SESSION = "33333333-3333-4333-8333-333333333333"
STAR_SESSION = "44444444-4444-4444-8444-444444444444"  # merchant "*"
UNKNOWN_SESSION = "55555555-5555-4555-8555-555555555555"  # no such row
ROUTE_RAN = 418

# Each merchant's own reseller, as the merchants table has it.
MERCHANT_RESELLERS = {
    MERCHANT: RESELLER,
    "other-merchant": "other-reseller",
    "victim-shop": "victim-reseller",
}

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
    LABELLED_SESSION: SimpleNamespace(
        id=LABELLED_SESSION,
        reseller_id=RESELLER,
        merchant_id="victim-shop",
        status=ChatSessionStatus.ACTIVE,
        metadata={"template_vars": {"rider_ref": "customer-3"}},
    ),
    STAR_SESSION: SimpleNamespace(
        id=STAR_SESSION,
        reseller_id=RESELLER,
        merchant_id="*",
        status=ChatSessionStatus.ACTIVE,
        metadata={"template_vars": {"rider_ref": "customer-4"}},
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
WILDCARD_LOGIN = _login(UserRole.RESELLER, ["*"], ["*"])
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

    async def merchant_lookup(merchant_id: str):
        if merchant_id not in MERCHANT_RESELLERS:
            return None
        return SimpleNamespace(reseller_id=MERCHANT_RESELLERS[merchant_id])

    reached = AsyncMock(side_effect=HTTPException(status_code=ROUTE_RAN))
    monkeypatch.setattr(handlers, "get_chat_session_by_id", lookup)
    # The gate uses the CRM doors' rule, which reads the merchants table.
    monkeypatch.setattr(
        crm_auth, "get_merchant_by_merchant_identifier", merchant_lookup
    )
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


@pytest.mark.parametrize("call", _calls(LABELLED_SESSION), ids=ROUTES)
def test_a_star_on_merchants_covers_only_the_tokens_resellers_merchants(
    client: TestClient, route_body: AsyncMock, call
) -> None:
    """The session carries the caller's reseller, but its merchant is another
    reseller's. A token with "*" on merchants and one reseller must not reach
    that merchant's riders."""
    resp = _send(client, call, RESELLER_LOGIN)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat session not found"
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls(STAR_SESSION), ids=ROUTES)
def test_a_merchant_called_star_is_nobodys(
    client: TestClient, route_body: AsyncMock, call
) -> None:
    """A wildcard token holds "*" itself: a session whose merchant is
    literally "*" must not match it."""
    resp = _send(client, call, RESELLER_LOGIN)
    assert resp.status_code == 404
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls(UNKNOWN_SESSION), ids=ROUTES)
def test_an_unknown_session_reads_like_one_out_of_scope(
    client: TestClient, route_body: AsyncMock, call
) -> None:
    resp = _send(client, call, MERCHANT_LOGIN)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat session not found"
    route_body.assert_not_awaited()


@pytest.mark.parametrize("call", _calls("not-a-uuid-1234"), ids=ROUTES)
@pytest.mark.parametrize(
    "session_id",
    ["not-a-uuid-1234", "{" + SESSION + "}", "urn:uuid:" + SESSION],
    ids=["not a uuid", "braces", "urn form"],
)
def test_a_malformed_session_id_is_an_unknown_session_not_a_500(
    client: TestClient, route_body: AsyncMock, monkeypatch, call, session_id: str
) -> None:
    """Session ids are UUIDs and the database refuses anything else, so the
    gate must not look a malformed one up."""
    looked_up = AsyncMock(return_value=None)
    monkeypatch.setattr(handlers, "get_chat_session_by_id", looked_up)
    method, path, data = call
    resp = _send(
        client, (method, path, {**data, "session_id": session_id}), MERCHANT_LOGIN
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat session not found"
    looked_up.assert_not_awaited()
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
    [MERCHANT_LOGIN, RESELLER_LOGIN, WILDCARD_LOGIN, ADMIN_LOGIN],
    ids=["the merchant", "its reseller", "star on both lists", "an admin"],
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
