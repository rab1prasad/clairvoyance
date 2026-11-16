"""Path-aware CORS dispatcher for the widget public surface.

Background — two CORS layers exist on this service:

1. **App-level FastAPI ``CORSMiddleware``** (``app/main.py``) — enforces
   the static ``CORS_ALLOWED_ORIGINS`` list. Right gate for admin /
   RBAC routes (portal-only surfaces) where the origin set is small
   and stable.

2. **Per-widget ``allowed_origins``** (``widget_common.py``
   :func:`resolve_widget_config_for_request`) — enforces the
   merchant-configured origin allowlist stored on the ``widget_config``
   row. The real production gate for the public widget surface;
   onboarding a new merchant means updating one DB row, not a redeploy.

The widget routes already mount their own ``@router.options(...)``
handlers that answer the preflight via :func:`options_cors_response`
so the merchant-side preflight always succeeds and reaches our
application logic. But the global ``CORSMiddleware`` runs first and
400s any preflight whose ``Origin`` isn't in the static list — so the
route-level OPTIONS handler is never reached, and adding a new
merchant to production currently requires editing the
``CORS_ALLOWED_ORIGINS`` env var + a rolling restart.

This middleware fixes that asymmetry: requests against the public
widget path prefix bypass the static-list middleware entirely (and
hit the route-level OPTIONS / per-widget allowlist instead), while
every other path stays behind the strict static list.

Which origin a widget answer names
----------------------------------

``Access-Control-Allow-Origin`` on a widget path is never ``*``. This
middleware is its only writer, and it names exactly one origin:

- **The origin validated for the widget the request is about**, when
  there is one. Either ``enforce_widget_origin`` just checked the
  caller against the row's ``allowed_origins`` (session create, the
  storefront config), or the request carries a widget token, which is
  bound to the origin its session was opened from
  (``widget_token.require_widget_session``). A page on any other origin
  that sends the same request gets an answer its browser will not let
  it read.
- **Otherwise the caller's own origin.** Only answers that hold nothing
  of a merchant's take this path: preflights (they carry no key and no
  token, so no widget is known yet), refusals, and the storefront
  loader's "no widget on this shop". The embed has to read those to
  know what to do next, from whatever shop it runs on.

This binds browsers only: a non-browser client ignores CORS and can
send any ``Origin`` it likes, which is why the allowlist check and the
token stay the authorization. Credentials stay off — widget tokens
travel in ``Authorization``, so cookies are never needed.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlparse

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Paths that bypass the global allowlist. Trailing slash is meaningful:
# ``/widget/`` matches the public session/message routes but not the
# admin-only ``/widget-config`` CRUD which must stay locked to portal
# origins.
_DEFAULT_BYPASS_PREFIXES: tuple[str, ...] = ("/agent/voice/breeze-buddy/widget/",)

# Where a request's validated origin is kept: ``request.state`` is backed
# by ``scope["state"]``, the one dict both the route and this middleware
# can reach.
_VALIDATED_ORIGIN_KEY = "widget_validated_origin"


def _origin_from_referer(referer: Optional[str]) -> Optional[str]:
    """Strip a Referer URL down to ``scheme://host[:port]`` for allowlist
    comparison. Returns None when the header is missing or unparseable.
    """
    if not referer:
        return None
    try:
        parsed = urlparse(referer)
    except Exception:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def caller_origin(headers: Mapping[str, str]) -> Optional[str]:
    """Pick the caller's origin: ``Origin`` header wins; ``Referer`` is a
    fallback for the rare cases where a browser strips Origin (legacy
    cross-origin GETs, some privacy proxies).

    One function for both the allowlist check and the CORS answer, so the
    two can never read a request's origin differently.
    """
    origin = headers.get("origin")
    if origin:
        return origin.strip() or None
    return _origin_from_referer(headers.get("referer"))


def remember_validated_origin(request: Request, origin: str) -> None:
    """Record the origin this request's widget was validated for. The CORS
    answer then names it instead of the caller's own origin."""
    setattr(request.state, _VALIDATED_ORIGIN_KEY, origin)


def validated_origin(request: Request) -> Optional[str]:
    """The origin recorded by :func:`remember_validated_origin`, if any."""
    return getattr(request.state, _VALIDATED_ORIGIN_KEY, None)


def _header_value(origin: Optional[str]) -> Optional[bytes]:
    if not origin:
        return None
    try:
        return origin.encode("latin-1")
    except UnicodeEncodeError:
        return None


class CustomWidgetCorsBypassMiddleware:
    """ASGI dispatcher: send widget paths straight through, everything
    else through ``CORSMiddleware``.

    The widget surface enforces origin allowlist per-merchant inside
    the route handlers (see ``widget_common.resolve_widget_config_for_request``)
    and emits its own preflight via ``@router.options(...)``. The
    static allowlist would only get in the way and force a redeploy
    for every new merchant onboarding, so widget paths bypass it.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allow_origins: Sequence[str],
        allow_credentials: bool = True,
        allow_methods: Iterable[str] = ("*",),
        allow_headers: Iterable[str] = ("*",),
        bypass_path_prefixes: Sequence[str] = _DEFAULT_BYPASS_PREFIXES,
    ) -> None:
        self._inner_app = app
        # Wrap the inner app in the strict CORS middleware. Non-bypass
        # paths get the same behaviour as before this middleware was
        # introduced — drop-in compatible.
        self._cors_app = CORSMiddleware(
            app,
            allow_origins=list(allow_origins),
            allow_credentials=allow_credentials,
            allow_methods=list(allow_methods),
            allow_headers=list(allow_headers),
        )
        self._bypass_prefixes = tuple(bypass_path_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Non-HTTP traffic (websockets, lifespan) doesn't touch CORS;
        # pass it through the same wrapped chain the strict middleware
        # uses so we don't accidentally drop lifespan events.
        if scope.get("type") != "http":
            await self._cors_app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(p) for p in self._bypass_prefixes):
            # Skip the static allowlist entirely. The widget routes
            # carry their own OPTIONS handlers AND validate the caller's
            # origin against widget_config.allowed_origins inside the
            # handler — that's the auth gate.
            #
            # Created here, before the route runs, so the dict the route
            # writes the validated origin into is the one read on the
            # way out.
            state = scope.setdefault("state", {})
            caller = caller_origin(Headers(scope=scope))
            await self._inner_app(scope, receive, self._wrap_send(send, state, caller))
            return

        await self._cors_app(scope, receive, send)

    @staticmethod
    def _wrap_send(
        send: Send, state: MutableMapping[str, Any], caller: Optional[str]
    ) -> Send:
        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Sole writer of the header on widget paths: whatever a
                # handler set is replaced, so no route can hand out ``*``.
                headers = [
                    (key, value)
                    for key, value in message.get("headers") or []
                    if key.lower() != b"access-control-allow-origin"
                ]
                origin = _header_value(state.get(_VALIDATED_ORIGIN_KEY) or caller)
                if origin is not None:
                    headers.append((b"access-control-allow-origin", origin))
                # The answer differs per origin, so a cache must key on it.
                if not any(
                    key.lower() == b"vary" and b"origin" in value.lower()
                    for key, value in headers
                ):
                    headers.append((b"vary", b"Origin"))
                message["headers"] = headers
            await send(message)

        return send_with_cors


__all__ = [
    "CustomWidgetCorsBypassMiddleware",
    "caller_origin",
    "remember_validated_origin",
    "validated_origin",
]
