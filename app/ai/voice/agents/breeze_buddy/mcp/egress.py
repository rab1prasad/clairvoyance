"""SSRF-safe egress for MCP servers.

Every MCP request, redirect hops included, is checked at send time and connects
to a checked address, so neither a redirect nor DNS rebinding reaches an
internal target.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import httpx2
import pipecat.services.mcp_service as pipecat_mcp
from mcp.client.session_group import StreamableHttpParameters
from mcp.client.streamable_http import streamable_http_client

from app.core.logger import logger
from app.core.network import ADDRESS_CONNECT_CAP, redact_url
from app.core.security.ssrf import (
    EgressResolutionError,
    SSRFError,
    is_same_origin,
    validate_egress_url,
)


async def _check_mcp_egress(url: str) -> List[str]:
    """Resolve the MCP server now and refuse internal targets; returns the
    checked IPs. The reason is logged, not raised: it reaches the LLM."""
    try:
        return await validate_egress_url(url)
    except EgressResolutionError:
        raise
    except SSRFError as e:
        logger.warning(f"[BUDDY_MCP] egress refused for {redact_url(url)}: {e}")
        raise SSRFError("MCP server blocked by egress policy") from None


class _EgressGuard:
    """Checks each request and connects to a checked address, not the name.

    One transport serves one MCP server: a hop to another host is refused, so
    the server's credentials and the tool arguments never leave it. The mcp
    library hangs on a raised error, so failures come back as a status instead.
    """

    # The HTTP library this transport plugs into; the check is the same on both.
    _http: Any = httpx

    def __init__(self) -> None:
        self._inner = self._http.AsyncHTTPTransport()
        self._origin: Optional[str] = None

    async def handle_async_request(self, request: Any) -> Any:
        http = self._http
        url = str(request.url)
        if self._origin is None:
            self._origin = url
        elif not is_same_origin(self._origin, url):
            logger.warning(
                f"[BUDDY_MCP] refused redirect {redact_url(self._origin)} -> "
                f"{redact_url(url)}"
            )
            return http.Response(404, request=request)
        try:
            budget = (request.extensions.get("timeout") or {}).get("connect")
            ips = await asyncio.wait_for(_check_mcp_egress(url), budget)
        except (asyncio.TimeoutError, EgressResolutionError) as e:
            logger.warning(f"[BUDDY_MCP] DNS failed for {redact_url(url)}: {e!r}")
            return http.Response(504, request=request)
        except SSRFError:
            return http.Response(404, request=request)  # reason logged by the check

        # raw_host is the ASCII (punycode) name the certificate carries.
        tls_name = request.url.raw_host.decode("ascii")

        def pinned(ip: str, last: bool) -> Any:
            # A copy: httpx joins relative redirects onto the caller's request.
            extensions: Dict[str, Any] = {
                **request.extensions,
                "sni_hostname": tls_name,
            }
            if not last:  # a dead address must not eat the whole budget
                timeout: Dict[str, Any] = dict(request.extensions.get("timeout") or {})
                connect = timeout.get("connect") or ADDRESS_CONNECT_CAP
                timeout["connect"] = min(connect, ADDRESS_CONNECT_CAP)
                extensions["timeout"] = timeout
            return http.Request(
                request.method,
                request.url.copy_with(host=ip),
                headers=request.headers,
                stream=request.stream,
                extensions=extensions,
            )

        for ip in ips[:-1]:
            try:
                return await self._inner.handle_async_request(pinned(ip, False))
            except (http.ConnectError, http.ConnectTimeout):
                continue  # never reached it, so the next address is safe
        return await self._inner.handle_async_request(pinned(ips[-1], True))

    async def aclose(self) -> None:
        await self._inner.aclose()


class McpEgressTransport(_EgressGuard, httpx.AsyncBaseTransport):
    """The guard on httpx: our direct tool calls."""


class McpSdkEgressTransport(_EgressGuard, httpx2.AsyncBaseTransport):
    """The guard on httpx2: the mcp library's own sessions (mcp 2 runs on httpx2)."""

    _http = httpx2


def _egress_safe_client(
    headers: Optional[dict] = None,
    timeout: Optional[httpx2.Timeout] = None,
    auth: Optional[httpx2.Auth] = None,
) -> httpx2.AsyncClient:
    """Same client the mcp library builds, sending through the egress check."""
    return httpx2.AsyncClient(
        follow_redirects=True,
        headers=headers,
        timeout=timeout or httpx2.Timeout(30.0),
        auth=auth,
        transport=McpSdkEgressTransport(),
    )


class McpServerParameters(StreamableHttpParameters):
    """Names the client an MCP session runs on; the opener below reads it."""

    httpx_client_factory: Any = _egress_safe_client


@asynccontextmanager
async def _guarded_streamable_http_transport(params: StreamableHttpParameters):
    """Open an MCP session on the guarded client.

    pipecat 1.14 builds the HTTP client itself and forwards none of our fields
    to the mcp library, so this takes the place of its opener (below): a session
    runs on the client its parameters name, the guarded one when they name none.
    """
    factory = getattr(params, "httpx_client_factory", None) or _egress_safe_client
    async with factory(
        headers=params.headers,
        timeout=httpx2.Timeout(params.timeout, read=params.sse_read_timeout),
    ) as client:
        async with streamable_http_client(
            params.url,
            http_client=client,
            terminate_on_close=params.terminate_on_close,
        ) as streams:
            yield streams


# pipecat's MCPClient opens every streamable-HTTP session through this name.
setattr(pipecat_mcp, "_streamable_http_transport", _guarded_streamable_http_transport)
