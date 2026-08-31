"""/crm/connectivity — the module's HTTP surface. Thin routes (rules §1).

Read the auth model here before adding a route, because this router
deliberately holds two kinds:

  · ``/installations/{id}/subscribe`` is ordinary admin work — merchant JWT
    via Depends, tenancy by query param, like every other CRM route.
  · ``/webhooks/whatsapp`` (GET and POST) carry **no Depends at all**. Meta
    cannot hold a bearer token, so those two authenticate each request by its
    signature, checked inside the handler before the body is read. That is
    the design, not an omission — and it is why auth must never be attached
    to this router as a whole. A blanket dependency here would lock Meta out
    and silently stop every event.

The webhook path names WhatsApp rather than taking a {channel} parameter,
because WhatsApp is the only provider that sends us callbacks. A second one
gets its own route beside this pair — which is also when we will know whether
it can share anything with them.

The handlers stay dumb on purpose. Verifying a callback and filing what it
carries lives in webhooks.py and the channel's adapter; these two only turn an
outcome into a status code, so the door is testable without HTTP.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.core.logger import logger
from app.core.logger.context import set_log_context
from app.crm.auth import crm_admin_user
from app.crm.connectivity.contracts import (
    OUTCOME_BAD_SIGNATURE,
    OUTCOME_UNREADABLE,
    SubscriptionRefused,
    ingest_whatsapp_webhook,
    subscribe_installation,
    whatsapp_handshake_challenge,
)
from app.schemas import UserInfo

router = APIRouter()

# Meta's webhook payloads are a few kilobytes even with batched entries; a
# megabyte is generous slack. The cap exists because the webhook route is
# unauthenticated BY DESIGN (a provider cannot hold a bearer token), so
# without it any caller could have us buffer an arbitrarily large body into
# memory before the signature check gets a chance to refuse them.
MAX_WEBHOOK_BODY_BYTES = 1024 * 1024


class SubscriptionResult(BaseModel):
    """What was subscribed. The provider's account id is echoed so an operator
    running this across several accounts can see which one answered."""

    installation_id: str
    external_account_id: str
    subscribed: bool = True


@router.post(
    "/installations/{installation_id}/subscribe",
    response_model=SubscriptionResult,
)
async def subscribe_installation_route(
    installation_id: str,
    merchant_id: str = Query(..., description="Tenant scope — required"),
    current_user: UserInfo = Depends(crm_admin_user),
) -> SubscriptionResult:
    """Subscribe this connected account to our app, whatever channel it is on.

    Until this runs, that merchant's events are never sent to us: registering
    a callback URL with the provider routes nothing on its own.

    409 rather than 400 on refusal — the request was understood and
    deliberately declined (no credentials, the provider said no), and the
    message says which so the caller can act on it.
    """
    set_log_context(component="crm.connectivity.subscribe", merchant_id=merchant_id)
    try:
        account_id = await subscribe_installation(merchant_id, installation_id)
    except SubscriptionRefused as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return SubscriptionResult(
        installation_id=installation_id, external_account_id=account_id or ""
    )


@router.get("/webhooks/whatsapp", response_class=PlainTextResponse)
async def whatsapp_handshake_route(request: Request) -> Response:
    """Meta's subscription challenge, echoed back once at registration.

    404 rather than 403 on refusal: answering differently for a wrong token
    than for a missing one tells an unauthenticated caller they have found
    something worth guessing at.
    """
    set_log_context(component="crm.connectivity.webhooks")
    challenge = whatsapp_handshake_challenge(request.query_params)
    if challenge is None:
        return PlainTextResponse("", status_code=status.HTTP_404_NOT_FOUND)
    logger.info("connectivity: whatsapp webhook subscription verified")
    # Echoed raw: Meta compares the body, so quoting or JSON-wrapping it fails
    # the handshake.
    return PlainTextResponse(challenge)


@router.post("/webhooks/whatsapp")
async def whatsapp_webhook_route(request: Request) -> Response:
    """Verify a Meta callback and file its letters in the event spine.

    The raw bytes are read before anything else touches them: the signature
    covers exactly what was sent, and parse-then-reserialise would change them
    enough to break it forever.

    200 means RECEIVED, never UNDERSTOOD. A letter filed but not yet
    interpreted is still received, and answering anything else would ask Meta
    to send it again — the one response that makes a bad moment worse.
    """
    set_log_context(component="crm.connectivity.webhooks")
    # The read itself is bounded, not just a declared Content-Length — a
    # chunked body has no header to check, and this is the one route that
    # must buffer bytes from callers it has not authenticated yet.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_WEBHOOK_BODY_BYTES:
            return Response(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    outcome = await ingest_whatsapp_webhook(bytes(body), request.headers)
    if outcome == OUTCOME_BAD_SIGNATURE:
        # No detail: a caller who cannot sign a request has not earned an
        # explanation of why theirs failed.
        return Response(status_code=status.HTTP_403_FORBIDDEN)
    if outcome == OUTCOME_UNREADABLE:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    return Response(status_code=status.HTTP_200_OK)
