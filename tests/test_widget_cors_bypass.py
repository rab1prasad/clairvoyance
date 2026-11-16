# pyrefly: ignore-errors
"""Tests for [[widget_cors_bypass.CustomWidgetCorsBypassMiddleware]].

Two things are checked:

  * the asymmetry: widget public paths
    (``/agent/voice/breeze-buddy/widget/…``) are NOT subject to the static
    ``CORS_ALLOWED_ORIGINS`` list — any origin can preflight; per-merchant
    ``widget_config.allowed_origins`` is the auth gate — while admin / RBAC
    paths still go through ``CORSMiddleware`` with the static list.
  * which origin a widget answer names: never ``*``. The origin the route
    validated when there is one, else the caller's own.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from app.api.routers.breeze_buddy.widget_common import options_cors_response
from app.core.middleware.widget_cors_bypass import (
    CustomWidgetCorsBypassMiddleware,
    remember_validated_origin,
)

WIDGET = "/agent/voice/breeze-buddy/widget"
STORE = "https://store.example"
OTHER = "https://other.example"
WIDGET_ROUTES = ["open", "validated", "refused", "star", "vary-origin", "vary-other"]


def _build_app() -> FastAPI:
    """Tiny FastAPI mirror: widget-shaped routes that stand for each kind of
    answer, and one admin-shaped route — to exercise both middleware
    branches without pulling in the full breeze-buddy router stack."""
    app = FastAPI()

    @app.get(f"{WIDGET}/open")
    def nothing_validated() -> dict:
        return {"ok": True}

    @app.get(f"{WIDGET}/validated")
    def store_validated(request: Request) -> dict:
        remember_validated_origin(request, STORE)
        return {"ok": True}

    @app.get(f"{WIDGET}/refused")
    def refused() -> dict:
        raise HTTPException(
            status_code=403, detail="Origin not permitted for this widget"
        )

    @app.get(f"{WIDGET}/star")
    def sets_its_own_wildcard() -> Response:
        return Response(headers={"Access-Control-Allow-Origin": "*"})

    @app.get(f"{WIDGET}/vary-origin")
    def already_varies_on_origin() -> Response:
        return Response(headers={"Vary": "Origin"})

    @app.get(f"{WIDGET}/vary-other")
    def varies_on_something_else() -> Response:
        return Response(headers={"Vary": "Accept-Encoding"})

    # The real preflight answer the widget routers mount.
    @app.options(f"{WIDGET}/open")
    def widget_options() -> Response:
        return options_cors_response()

    @app.get("/agent/voice/breeze-buddy/templates")
    def admin_get() -> dict:
        return {"ok": True}

    app.add_middleware(
        CustomWidgetCorsBypassMiddleware,
        allow_origins=["https://portal.breeze.in"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_build_app())


def _allow_origin(resp) -> list[str]:
    """Every Access-Control-Allow-Origin value on the response (httpx would
    join duplicates with ``, `` and hide them)."""
    return resp.headers.get_list("access-control-allow-origin")


# -- widget paths: which origin the answer names ---------------------------


def test_widget_preflight_allowed_from_arbitrary_origin(client: TestClient) -> None:
    """The whole point of the bypass: a brand-new merchant origin must
    reach our handler so per-widget allowlist can run. Without the
    bypass the global allowlist would 400 here.

    The answer names that origin, once — not ``*``.
    """
    resp = client.options(
        f"{WIDGET}/open",
        headers={
            "Origin": "https://firstbud.in",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 204
    assert _allow_origin(resp) == ["https://firstbud.in"]
    assert "POST" in resp.headers["access-control-allow-methods"]
    assert "authorization" in resp.headers["access-control-allow-headers"]


def test_an_answer_with_nothing_validated_names_the_caller(client: TestClient) -> None:
    """Without the header the browser would block JS from reading the body
    even though the request reached our handler. The bypass middleware has
    to add it because we skip ``CORSMiddleware`` and the widget handlers
    don't set CORS headers themselves (they return Pydantic models)."""
    resp = client.get(f"{WIDGET}/open", headers={"Origin": "https://firstbud.in"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert _allow_origin(resp) == ["https://firstbud.in"]
    assert "Origin" in resp.headers.get("vary", "")


def test_an_answer_names_the_validated_origin_not_the_caller(
    client: TestClient,
) -> None:
    """A page on another site gets an answer naming the store's origin, so
    its browser refuses to hand it the body."""
    resp = client.get(f"{WIDGET}/validated", headers={"Origin": OTHER})
    assert resp.status_code == 200
    assert _allow_origin(resp) == [STORE]


def test_the_validated_origin_reads_its_own_answer(client: TestClient) -> None:
    resp = client.get(f"{WIDGET}/validated", headers={"Origin": STORE})
    assert _allow_origin(resp) == [STORE]


def test_a_refusal_can_be_read_by_the_caller(client: TestClient) -> None:
    """The embed has to see the 403 to know its origin is not allowed."""
    resp = client.get(f"{WIDGET}/refused", headers={"Origin": OTHER})
    assert resp.status_code == 403
    assert _allow_origin(resp) == [OTHER]


def test_a_wildcard_set_by_a_handler_is_replaced(client: TestClient) -> None:
    resp = client.get(f"{WIDGET}/star", headers={"Origin": OTHER})
    assert _allow_origin(resp) == [OTHER]


@pytest.mark.parametrize("route", WIDGET_ROUTES)
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": STORE},
        {"Origin": OTHER},
        {"Origin": "null"},
        {"Referer": f"{OTHER}/products/1"},
        {},
    ],
    ids=["store", "other site", "null origin", "referer only", "no origin"],
)
def test_no_widget_answer_is_a_wildcard_or_allows_credentials(
    client: TestClient, route: str, headers: dict
) -> None:
    resp = client.get(f"{WIDGET}/{route}", headers=headers)
    assert "*" not in _allow_origin(resp)
    assert len(_allow_origin(resp)) <= 1
    assert "access-control-allow-credentials" not in resp.headers


def test_no_origin_and_no_referer_gets_no_allow_origin(client: TestClient) -> None:
    """A caller that is not a browser page: nothing to name."""
    resp = client.get(f"{WIDGET}/open")
    assert resp.status_code == 200
    assert _allow_origin(resp) == []


def test_a_missing_origin_falls_back_to_the_referer(client: TestClient) -> None:
    """Same fallback the allowlist check uses, so a browser whose Origin was
    stripped on the way keeps working."""
    resp = client.get(
        f"{WIDGET}/open", headers={"Referer": "https://firstbud.in/products/1?x=1"}
    )
    assert _allow_origin(resp) == ["https://firstbud.in"]


def test_vary_origin_is_added_once(client: TestClient) -> None:
    resp = client.get(f"{WIDGET}/vary-origin", headers={"Origin": STORE})
    assert resp.headers.get_list("vary") == ["Origin"]


def test_vary_origin_is_added_next_to_another_vary(client: TestClient) -> None:
    resp = client.get(f"{WIDGET}/vary-other", headers={"Origin": STORE})
    assert resp.headers.get_list("vary") == ["Accept-Encoding", "Origin"]


# -- everything else stays behind the static list --------------------------


def test_admin_preflight_rejected_from_unknown_origin(client: TestClient) -> None:
    """Admin / RBAC paths must STILL be gated by the static list — the
    bypass is widget-only. portal.breeze.in is allowed; everything else
    should 400 on preflight."""
    resp = client.options(
        "/agent/voice/breeze-buddy/templates",
        headers={
            "Origin": "https://firstbud.in",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette's CORSMiddleware returns 400 for disallowed origins
    # on preflight, no allow-origin header echoed.
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_admin_preflight_allowed_from_listed_origin(client: TestClient) -> None:
    """The static list still works for the origins it's meant to
    protect (portal/admin tools)."""
    resp = client.options(
        "/agent/voice/breeze-buddy/templates",
        headers={
            "Origin": "https://portal.breeze.in",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://portal.breeze.in"


def test_widget_config_admin_path_stays_strict() -> None:
    """Trailing-slash discipline: ``/widget-config`` (admin CRUD) must
    NOT be bypassed — only ``/widget/`` (note the slash) is the public
    surface. This guards against accidentally widening the bypass to
    the admin widget-config endpoints if someone trims the prefix."""
    app = FastAPI()

    @app.get("/agent/voice/breeze-buddy/widget-config")
    def widget_config() -> dict:
        return {"ok": True}

    app.add_middleware(
        CustomWidgetCorsBypassMiddleware,
        allow_origins=["https://portal.breeze.in"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    client = TestClient(app)
    resp = client.options(
        "/agent/voice/breeze-buddy/widget-config",
        headers={
            "Origin": "https://firstbud.in",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 400
