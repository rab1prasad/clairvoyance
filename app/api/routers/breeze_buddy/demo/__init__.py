"""
Public demo endpoint for Breeze Buddy (Daily web sessions).

Thin router that delegates to handler logic in handlers.py to follow
existing Breeze Buddy router conventions.
"""

from typing import Any, Dict

from fastapi import APIRouter, Request

from .handlers import DemoConnectRequest, breeze_buddy_demo_connect_handler

# No login on this route, so a page on another site could post to it from a
# visitor's browser. A JSON body with no Content-Type needs no preflight; the
# app parses such a body (main.py), this router does not.
router = APIRouter(strict_content_type=True)


@router.post("/demo/connect", tags=["demo"])
async def breeze_buddy_demo_connect(
    request: Request, body: DemoConnectRequest
) -> Dict[str, Any]:
    """Public demo connect endpoint (no auth)."""
    return await breeze_buddy_demo_connect_handler(request, body)
