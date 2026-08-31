"""WhatsApp, straight at Meta's Cloud API — no aggregator in between.

The first child of ChannelAdapter and the shape every later one copies. This
file owns EVERYTHING Meta-shaped, in both directions:

  outbound  build a request from the manifest row, read the answer, classify
            it — without deciding whether to retry, only whether retrying
            could plausibly differ;
  inbound   verify their signature, answer their handshake, unwrap their
            entry/changes/value envelope into letters, and subscribe an
            account so those callbacks start arriving.

None of the inbound half is general. Another provider signs a different way
over different bytes and nests its events differently, which is exactly why
it lives here rather than in the module's shared code.

Sends are template-only by design, not by omission: the manifest stores
template_id + variables and never a rendered string, because Meta renders the
final text and our own copy would be a guess at what the customer saw.
Free-form text needs an open conversation, which is the conversations
module's job.

Credential bundle (written by onboarding, read here):
    system_user_token   the bearer for every call        [required]
    app_secret          verifies inbound webhooks        [phase 3]
    verify_token        the webhook handshake secret     [phase 3]
"""

import hashlib
import hmac
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import quote

import httpx

from app.core.config.static import (
    CRM_MESSAGE_SEND_TIMEOUT_SECONDS,
    META_APP_SECRET,
    META_WEBHOOK_VERIFY_TOKEN,
    META_WHATSAPP_GRAPH_BASE_URL,
    META_WHATSAPP_GRAPH_VERSION,
)
from app.core.logger import logger
from app.core.transport.http_client import create_http_client
from app.crm.connectivity.providers.base import ChannelAdapter, require_secret
from app.crm.connectivity.reasons import (
    REASON_BAD_ADDRESS,
    REASON_BAD_VARIABLES,
    REASON_NO_CREDENTIAL,
    REASON_NO_TEMPLATE,
    REASON_UNREADABLE,
)
from app.crm.connectivity.schemas import (
    TOPIC_INBOUND,
    TOPIC_STATUS,
    ChannelBinding,
    CredentialBundle,
    InboundLetter,
    QueuedMessage,
    SendOutcome,
)
from app.crm.shared.redact import mask_address, mask_digit_runs

TOKEN_KEY = "system_user_token"

# Meta signs every callback with the APP secret over the raw request body.
# Platform-level, not per-merchant, and that is forced rather than chosen: the
# payload naming the merchant cannot be trusted until the signature is
# verified, and verifying it needs the secret.
SIGNATURE_HEADER = "x-hub-signature-256"
_SIGNATURE_PREFIX = "sha256="

# Meta's error codes, split by the only question an adapter may ask: could
# the same request plausibly succeed later?
#
# Retryable — the provider is busy or pacing us, not refusing on the merits.
# The first three are Graph/app/WABA throttles that arrive as HTTP 400, which
# the unknown-4xx default below would read as terminal — permanently failing
# every message queued during a throttle window instead of backing off.
RETRYABLE_CODES = {
    "4",  # app-level "API Too Many Calls"
    "613",  # Graph rate limit exceeded
    "80007",  # WABA rate limit
    "130429",  # Cloud API throughput limit
    "131048",  # spam rate limit
    "131049",  # per-user engagement limit ("healthy ecosystem")
    "131056",  # business/consumer pair rate limit
}

# Terminal — waiting changes nothing; retrying just collects the identical
# refusal three times.
TERMINAL_CODES = {
    "100",  # invalid parameter
    "131008",  # required parameter missing
    "131009",  # parameter value invalid
    "131026",  # undeliverable: recipient cannot receive WhatsApp messages
    "131047",  # 24-hour window closed — a template is the fix, not a retry
    "132000",  # template param count mismatch
    "132001",  # template does not exist
    "132005",  # rendered template too long
    "132007",  # template content policy violation
    "132012",  # template parameter format mismatch
    "132015",  # template paused
    "132016",  # template disabled
}

# Terminal, and a statement about the CONNECTION rather than this message:
# every queued message for that merchant is about to fail the same way. The
# send path does not act on them (that is channel-lifecycle work, not built
# yet) — they are named so that module has an exact signal to watch for on
# crm_message.reason instead of guessing which codes mean "re-authenticate".
CREDENTIAL_CODES = {
    "10",  # permission denied
    "190",  # invalid or expired access token
    "200",  # permissions error
    "133010",  # phone number not registered for Cloud API
}


class WebhookSubscriptionError(RuntimeError):
    """Meta refused to route a merchant's callbacks to us. Raised rather than
    returning a flag: the caller is a request handler that must tell the
    merchant it did not work, and a subscription that silently failed looks
    exactly like a healthy account until somebody wonders why no events ever
    arrive."""


_NON_DIGITS = re.compile(r"\D")


def to_meta_recipient(address: str) -> Optional[str]:
    """E.164 in, Meta's digits-only form out.

    Stripping happens HERE and the stripped form is never persisted: one
    representation in the database, whatever each provider prefers at its
    own edge.
    """
    digits = _NON_DIGITS.sub("", address or "")
    # Deliberate parity with shared/normalize.py's ^\+[1-9][0-9]{6,14}$ (and
    # the platform_identity CHECK), so a number this system was willing to
    # store is never rejected here as an "invalid address". 15 is E.164's
    # ceiling; 7 is the real short end (Saint Helena, +290 plus 4 digits);
    # no country code starts with 0.
    if not 7 <= len(digits) <= 15 or digits.startswith("0"):
        return None
    return digits


# The value types str() renders faithfully. bool is refused below despite
# being an int subclass: str(True) is 'True', which no customer message
# means to say.
_TEXTABLE_TYPES = (str, int, float)


def build_parameters(variables: Dict[str, Any]) -> Union[List[Dict[str, Any]], str]:
    """Manifest variables -> Meta template body parameters, or the defect.

    Meta accepts two forms and the producer chooses by how it writes the keys:

      {"1": "Priya", "2": "ORD-42"}         -> positional, in numeric order
      {"customer_name": "Priya", ...}       -> named (parameter_name)

    A str return means the dict cannot be sent and says why — the caller
    logs it and refuses terminally with REASON_BAD_VARIABLES. Two defects
    earn that:

      · A value that is not text or a number. str() rendered a JSON null as
        the literal word 'None' inside a customer's message — corruption
        that LOOKS delivered. The defect names the key and type, never the
        value, which may be personal data.
      · Positional and named keys mixed. Meta takes one style per request,
        so no rendering is correct; guessing one only buys a round trip to
        the refusal this string already states.

    ASCII digits decide positional vs named, not str.isdigit(), which also
    accepts digit-CATEGORY characters like '²' that int() then refuses —
    turning a bad key into a mid-send exception instead of an outcome.
    """
    if not variables:
        return []
    items = [(str(key), value) for key, value in variables.items()]
    for key, value in items:
        if isinstance(value, bool) or not isinstance(value, _TEXTABLE_TYPES):
            return f"variable '{key}' is {type(value).__name__}, not text"
    positional = [key for key, _ in items if key.isascii() and key.isdigit()]
    if len(positional) == len(items):
        # The keys must be exactly 1..N: Meta fills {{1}}..{{N}} from list
        # ORDER, so a gap ({"1","3"}) or an off-origin key ({"2"} alone,
        # {"0","1"}) silently shifts every later value one slot left in a
        # customer's message — corruption that LOOKS delivered, like the
        # bool case above. (Duplicates cannot happen: keys came from a dict.)
        if sorted(int(key) for key in positional) != list(range(1, len(items) + 1)):
            return "positional template variable keys must be exactly 1..N"
        # Sorting as strings would put "10" before "2" and silently swap two
        # values in a customer's message.
        ordered = sorted(items, key=lambda item: int(item[0]))
        return [{"type": "text", "text": str(value)} for _, value in ordered]
    if positional:
        return "mixes positional and named template variables"
    return [
        {"type": "text", "parameter_name": key, "text": str(value)}
        for key, value in items
    ]


# ---------------------------------------------------------------------------
# Inbound: Meta's callback shape
#
# Everything below is Meta-specific and exists nowhere else in the codebase.
# A second provider signs differently and nests differently, and that is the
# whole reason this lives behind the adapter interface.
# ---------------------------------------------------------------------------


def provider_timestamp(value: Any) -> Optional[datetime]:
    """Meta's unix seconds -> an aware datetime, or None if unusable.

    Their clock, not ours: when something happened is the provider's fact,
    and when we heard about it is merely ours. Total, because a letter with a
    broken timestamp is still worth filing.
    """
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def envelope_values(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The payload objects inside one notification.

    Meta wraps everything in entry[] / changes[] / value and batches freely:
    one body may hold several entries, each with several changes, and they may
    concern different merchants' numbers. Total, so a malformed entry cannot
    discard a sibling that was perfectly good.
    """
    values: List[Dict[str, Any]] = []
    entries = body.get("entry")
    if not isinstance(entries, list):
        return values
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if isinstance(value, dict):
                values.append(value)
    return values


def receiving_number(value: Dict[str, Any]) -> str:
    """Which of OUR numbers a notification concerns.

    Meta names it as a phone_number_id in the metadata, and it is the only
    thing in the body that can say whose customer this is — so a value
    without one cannot be filed under any merchant.
    """
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("phone_number_id") or "")
    return ""


def letters_in_value(value: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """One notification value -> (topic, external_id, payload) per item.

    Two kinds ride the same notification and both are wanted:

      · statuses[] — what became of a message WE sent. Meta sends one per
        transition on the same message id, so the id alone would collapse
        four letters into one; the external_id pairs it with the status.
      · messages[] — what a customer sent US. Its own id is unique already.

    The payload is the provider's object as it arrived, with one key added:
    the notification's ``metadata``, copied verbatim, because it names the
    receiving number and the item itself does not.
    """
    metadata = value.get("metadata")
    letters: List[Tuple[str, str, Dict[str, Any]]] = []

    statuses = value.get("statuses")
    if isinstance(statuses, list):
        for status in statuses:
            if not isinstance(status, dict):
                continue
            message_id, state = status.get("id"), status.get("status")
            if not message_id or not state:
                continue
            letters.append(
                (
                    TOPIC_STATUS,
                    f"{message_id}:{state}",
                    {**status, "metadata": metadata},
                )
            )

    messages = value.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            message_id = message.get("id")
            if not message_id:
                continue
            letters.append(
                (TOPIC_INBOUND, str(message_id), {**message, "metadata": metadata})
            )

    return letters


class MetaWhatsAppAdapter(ChannelAdapter):
    """Meta Cloud API, one merchant's phone number at a time."""

    channel = "whatsapp"

    def __init__(
        self,
        base_url: str = META_WHATSAPP_GRAPH_BASE_URL,
        api_version: str = META_WHATSAPP_GRAPH_VERSION,
    ) -> None:
        """Both are dials so a local run can point at a stub and exercise
        the whole dispatcher without sending anything to Meta."""
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version.strip("/")

    def endpoint(self, phone_number_id: str) -> str:
        """The per-number /messages URL — one endpoint per binding address."""
        # quote(..., safe="") pins the address inside one path segment. The
        # column has no format CHECK and no writer validates it: a '/' in a
        # bad row must not become URL structure carrying the bearer token to
        # another Graph path, and a control character must not raise
        # httpx.InvalidURL — not an HTTPError, so it sails past the catch.
        return (
            f"{self._base_url}/{self._api_version}/"
            f"{quote(phone_number_id, safe='')}/messages"
        )

    async def deliver(
        self,
        message: QueuedMessage,
        route_bundle: CredentialBundle,
        binding: ChannelBinding,
    ) -> SendOutcome:
        """Build the request, post it, classify the answer.

        Every refusal before the network is 'blocked', never 'failed':
        nothing was posted, so these are OUR refusals — 'failed' is the word
        the manifest reserves for the provider's no (T16 col 12). Terminal
        either way: a missing token, template or usable address does not
        change by retrying.
        """
        token = require_secret(route_bundle, TOKEN_KEY, self.channel)
        if token is None:
            # This reason must carry the same status here as it does from
            # resolve_send_route — one word, one meaning on the manifest.
            return SendOutcome(status="blocked", reason=REASON_NO_CREDENTIAL)

        if not message.template_id:
            # Not a retry: this row can never be sent as it stands.
            logger.error(f"whatsapp: message {message.id} has no template_id")
            return SendOutcome(status="blocked", reason=REASON_NO_TEMPLATE)

        recipient = to_meta_recipient(message.sent_to_address)
        if recipient is None:
            logger.error(
                f"whatsapp: message {message.id} address "
                f"{mask_address(message.sent_to_address, self.channel)} "
                f"is not a usable number"
            )
            return SendOutcome(status="blocked", reason=REASON_BAD_ADDRESS)

        parameters = build_parameters(message.variables)
        if isinstance(parameters, str):
            # No rendering of these variables is the right one; refuse here
            # instead of shipping a guess or letting Meta refuse one.
            logger.error(
                f"whatsapp: message {message.id} has unsendable variables — "
                f"{parameters}"
            )
            return SendOutcome(status="blocked", reason=REASON_BAD_VARIABLES)

        payload = self.build_payload(message, recipient, binding, parameters)
        url = self.endpoint(binding.address)

        try:
            async with create_http_client(
                timeout=CRM_MESSAGE_SEND_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as e:
            return self.transport_failure(e, message.sent_to_address)

        return self.read_response(response, message)

    def build_payload(
        self,
        message: QueuedMessage,
        recipient: str,
        binding: ChannelBinding,
        parameters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """The Cloud API send body. Assembly only — ``parameters`` arrive
        already built and judged sendable by deliver().

        Language comes from the binding: which locale a merchant's template
        is approved in is a per-endpoint fact, not a global setting. INTERIM
        until T23's registry keys templates by (merchant, channel, name,
        language) — then the registry decides and this read goes away.
        """
        language = str(binding.capabilities.get("template_language") or "en_US")
        components: List[Dict[str, Any]] = []
        if parameters:
            components.append({"type": "body", "parameters": parameters})
        return {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "template",
            "template": {
                "name": message.template_id,
                "language": {"code": language},
                "components": components,
            },
        }

    def read_response(
        self, response: httpx.Response, message: QueuedMessage
    ) -> SendOutcome:
        """Meta's answer -> SendOutcome. Classification only."""
        body = self.json_body(response)

        if response.is_success:
            provider_message_id = self.message_id_of(body)
            if provider_message_id is None:
                # Still 'accepted': Meta took it, and calling this a failure
                # would retry a message the customer may already have.
                logger.warning(
                    f"whatsapp: message {message.id} accepted without a wamid"
                )
            return SendOutcome(
                status="accepted", provider_message_id=provider_message_id
            )

        code, detail = self.error_of(body)
        # detail is Meta's text, not ours: their catalog strings carry no
        # values today, but a string we don't control could someday echo the
        # recipient — masking beats trusting the contract to hold.
        logger.warning(
            f"whatsapp: message {message.id} refused — "
            f"http={response.status_code} code={code or 'none'} "
            f"{mask_digit_runs(detail)}"
        )

        # Both classes are terminal for THIS row and behave identically here;
        # they stay separate sets because they differ in what they say about
        # the CONNECTION, which channel-lifecycle code reads off `reason`.
        if code in TERMINAL_CODES or code in CREDENTIAL_CODES:
            return SendOutcome(status="failed", reason=code)
        if code in RETRYABLE_CODES or response.status_code == 429:
            return SendOutcome(status="failed", reason=code or "429", retryable=True)

        # Unknown code: 5xx is Meta's problem and may pass, 4xx is ours and
        # will not — retrying an unknown 4xx spends attempts learning nothing.
        retryable = response.status_code >= 500
        return SendOutcome(
            status="failed",
            reason=code or f"http_{response.status_code}",
            retryable=retryable,
        )

    @staticmethod
    def message_id_of(body: Dict[str, Any]) -> Optional[str]:
        """The wamid, which every delivery receipt will be keyed by."""
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("id"):
                return str(first["id"])
        return None

    @staticmethod
    def error_of(body: Dict[str, Any]) -> tuple:
        """(code, human detail) from Meta's error envelope.

        The code lands in `reason` verbatim — the provider's own word, not
        our paraphrase, so "why?" has an answer matching Meta's docs.
        """
        error = body.get("error")
        if not isinstance(error, dict):
            return None, REASON_UNREADABLE
        code = error.get("code")
        return (
            str(code) if code is not None else None,
            str(error.get("message") or ""),
        )


async def subscribe_to_webhooks(waba_id: str, access_token: str) -> None:
    """Point a merchant's WABA at our app's callback URL.

    ``POST /{waba_id}/subscribed_apps``. Configuring the callback URL in the
    Meta app dashboard routes NOTHING on its own: until each WABA is
    subscribed to the app, that merchant's delivery receipts and inbound
    messages are simply never sent to us. This is the call that turns it on,
    once per account.

    Raises on failure rather than returning a flag: the caller is a request
    handler that must tell the merchant it did not work, and a subscription
    that silently failed looks exactly like a connected account until someone
    wonders why no receipts ever arrive.

    Signature matches the stub the Embedded Signup work (PR #1038) left for
    this, so its onboarding sequence can call it unchanged once both land.
    """
    url = (
        f"{META_WHATSAPP_GRAPH_BASE_URL.rstrip('/')}/"
        f"{META_WHATSAPP_GRAPH_VERSION.strip('/')}/"
        f"{quote(waba_id, safe='')}/subscribed_apps"
    )
    async with create_http_client(timeout=CRM_MESSAGE_SEND_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url, headers={"Authorization": f"Bearer {access_token}"}
        )
    if not response.is_success:
        code, detail = MetaWhatsAppAdapter.error_of(
            MetaWhatsAppAdapter.json_body(response)
        )
        raise WebhookSubscriptionError(
            f"Meta refused the webhook subscription for this account "
            f"(http={response.status_code} code={code or 'none'}): {detail}"
        )


def verify_signature(raw_body: bytes, headers: Mapping[str, str]) -> bool:
    """Whether this callback really came from Meta.

    HMAC-SHA256 over the RAW bytes, keyed by the app secret. Meta cannot hold
    a bearer token, so this IS the authentication for the callback route.

    Fails closed on every uncertainty — no secret configured, no header, a
    header in an unexpected shape. An endpoint that accepts unverifiable
    bodies is one anyone can write events into.

    Two details are load-bearing rather than stylistic: the MAC covers the raw
    bytes BEFORE any parse (re-serialising changes whitespace and key order,
    and the signature would never match again), and compare_digest is used
    instead of == (string comparison returns early on the first wrong
    character, and that timing difference is enough to let an attacker
    discover a valid signature byte by byte).
    """
    if not META_APP_SECRET:
        # Loud, because this is the difference between "authenticated" and
        # "open to the internet", and the symptom otherwise is silence.
        logger.error(
            "whatsapp: no Meta app secret configured — refusing every inbound "
            "webhook"
        )
        return False
    # Header names are case-insensitive on the wire; a plain dict of them is
    # not, so look the value up without assuming the sender's casing.
    header = next(
        (v for k, v in headers.items() if k.lower() == SIGNATURE_HEADER), None
    )
    if not header or not header.startswith(_SIGNATURE_PREFIX):
        return False
    expected = hmac.new(META_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len(_SIGNATURE_PREFIX) :])


def handshake_challenge(params: Mapping[str, str]) -> Optional[str]:
    """Meta's subscription challenge, echoed back when the token matches.

    Called once, when the callback URL is saved in the app dashboard, to prove
    we own the endpoint. Same fail-closed posture and constant-time compare as
    the signature: the verify token is a shared secret, and leaking it through
    timing would let someone else claim our callback URL.
    """
    if not META_WEBHOOK_VERIFY_TOKEN:
        logger.error(
            "whatsapp: no webhook verify token configured — refusing the "
            "subscription handshake"
        )
        return None
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if params.get("hub.mode") != "subscribe" or not token or challenge is None:
        return None
    if not hmac.compare_digest(token, META_WEBHOOK_VERIFY_TOKEN):
        logger.warning("whatsapp: webhook handshake presented a wrong token")
        return None
    return challenge


def read_notification(body: Dict[str, Any]) -> List[InboundLetter]:
    """Meta's envelope -> letters, ready for the spine.

    Total on purpose: a body this cannot read yields no letters rather than
    raising. Meta is owed a 200 either way, and a malformed fragment must not
    discard the good ones beside it.

    A value with no receiving number yields nothing: template-status and
    account-review notifications arrive on this same webhook, and they name no
    endpoint to file them under.
    """
    letters: List[InboundLetter] = []
    for value in envelope_values(body):
        address = receiving_number(value)
        if not address:
            continue
        for topic, external_id, payload in letters_in_value(value):
            letters.append(
                InboundLetter(
                    address=address,
                    topic=topic,
                    external_id=external_id,
                    payload=payload,
                    occurred_at=provider_timestamp(payload.get("timestamp")),
                )
            )
    return letters


async def subscribe_account(waba_id: str, bundle: CredentialBundle) -> None:
    """Subscribe one WABA using that account's own stored token.

    Takes the bundle rather than a token so the key's name stays in this file:
    which secret Meta wants is Meta's business, not the caller's.
    """
    token = require_secret(bundle, TOKEN_KEY, MetaWhatsAppAdapter.channel)
    if token is None:
        raise WebhookSubscriptionError(
            "This account's access token is missing or unreadable. "
            "Reconnect it first."
        )
    await subscribe_to_webhooks(waba_id, token)
