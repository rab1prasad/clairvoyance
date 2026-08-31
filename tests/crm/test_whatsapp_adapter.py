"""The WhatsApp adapter: what it posts, and what it makes of the answer.

No network anywhere. httpx.MockTransport stands in for Meta, so the entire
error matrix — including the ones that are painful to provoke for real, like
an expired token — is exercised on every test run.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pytest

from app.crm.connectivity.providers import whatsapp as whatsapp_module
from app.crm.connectivity.providers.whatsapp import (
    CREDENTIAL_CODES,
    RETRYABLE_CODES,
    TERMINAL_CODES,
    MetaWhatsAppAdapter,
    build_parameters,
    handshake_challenge,
    read_notification,
    to_meta_recipient,
    verify_signature,
)
from app.crm.connectivity.schemas import (
    ChannelBinding,
    CredentialBundle,
    QueuedMessage,
)

ACCEPTED_BODY = {
    "messaging_product": "whatsapp",
    "contacts": [{"input": "919876543210", "wa_id": "919876543210"}],
    "messages": [{"id": "wamid.HBgMOTE5ODc2NTQzMjEw"}],
}


def _message(**overrides) -> QueuedMessage:
    """A queued message for tests; keyword overrides replace any field."""
    fields = dict(
        id="m-1",
        merchant_id="shop",
        customer_id="c-1",
        channel="whatsapp",
        sent_to_address="+919876543210",
        source_kind="transactional",
        purpose_key="order_update",
        template_id="order_update_v1",
        variables={"1": "Priya", "2": "ORD-42"},
        dedupe_key="evt-1",
        attempt=1,
        next_attempt_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return QueuedMessage(**fields)


def _binding(**overrides) -> ChannelBinding:
    """An active channel binding for tests; overrides replace any field."""
    fields = dict(
        id="b-1",
        merchant_id="shop",
        channel="whatsapp",
        installation_id="i-1",
        address="PHONE_NUMBER_ID",
        capabilities={},
        is_primary=True,
        status="active",
    )
    fields.update(overrides)
    return ChannelBinding(**fields)


def _bundle(**values) -> CredentialBundle:
    """A credential bundle holding a usable token."""
    return CredentialBundle(values={"system_user_token": "tok", **values})


def _mocked(monkeypatch, handler) -> Dict[str, Any]:
    """Point the adapter's HTTP client at a canned responder and capture the
    request it made."""
    seen: Dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        """Record the outgoing request, then delegate to the handler."""
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read().decode()
        return handler(request)

    monkeypatch.setattr(
        whatsapp_module,
        "create_http_client",
        lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(_capture)),
    )
    return seen


def _responds(status: int, body: Optional[dict] = None, text: Optional[str] = None):
    """A canned HTTP responder with the given status and body."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Test double: canned provider response."""
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=body or {})

    return handler


async def _deliver(monkeypatch, handler, message=None, binding=None, bundle=None):
    """Run deliver() against a mocked transport; return (outcome, request seen)."""
    seen = _mocked(monkeypatch, handler)
    outcome = await MetaWhatsAppAdapter().deliver(
        message or _message(), bundle or _bundle(), binding or _binding()
    )
    return outcome, seen


# --- the request ------------------------------------------------------------


async def test_the_send_goes_to_this_bindings_number(monkeypatch) -> None:
    """The send goes to this bindings number."""
    # The endpoint is per-endpoint, not per-merchant: two numbers under one
    # account must not share a URL.
    _, seen = await _deliver(monkeypatch, _responds(200, ACCEPTED_BODY))
    assert seen["url"].endswith("/PHONE_NUMBER_ID/messages")
    assert seen["headers"]["authorization"] == "Bearer tok"


async def test_the_recipient_is_sent_without_its_plus(monkeypatch) -> None:
    """The recipient is sent without its plus."""
    # Stored E.164, posted Meta-style. The stripped form is never persisted.
    _, seen = await _deliver(monkeypatch, _responds(200, ACCEPTED_BODY))
    assert '"to":"919876543210"' in seen["body"].replace(" ", "")


async def test_the_body_names_a_template_and_never_a_rendered_string(
    monkeypatch,
) -> None:
    """The body names a template and never a rendered string."""
    _, seen = await _deliver(monkeypatch, _responds(200, ACCEPTED_BODY))
    body = seen["body"].replace(" ", "")
    assert '"type":"template"' in body
    assert '"name":"order_update_v1"' in body
    # Values are posted as parameters; we never assemble the sentence.
    assert '"text":"Priya"' in body


def test_numeric_keys_become_positional_parameters_in_numeric_order() -> None:
    """Numeric keys become positional parameters in numeric order."""
    # Sorting as strings would put "10" before "2" and silently swap two
    # values in a customer's message. Ten keys, because the string-vs-int
    # difference only shows once a two-digit key exists — and the keys must
    # be contiguous 1..N to be sendable at all (see the tests below).
    shuffled = {str(n): f"v{n}" for n in (2, 10, 1, 7, 4, 9, 3, 6, 8, 5)}
    params = build_parameters(shuffled)
    assert isinstance(params, list)
    assert [p["text"] for p in params] == [f"v{n}" for n in range(1, 11)]
    assert all("parameter_name" not in p for p in params)


def test_non_contiguous_positional_keys_are_refused() -> None:
    """Non-contiguous positional keys are refused."""
    # Meta fills {{1}}..{{N}} from list ORDER, not from the keys — so a gap
    # or an off-origin key silently shifts every later value one slot left
    # in the customer's message. Corruption that LOOKS delivered; refusing
    # is the only honest answer.
    for variables in (
        {"1": "a", "3": "c"},  # gap: "c" would render as {{2}}
        {"0": "a", "1": "b"},  # zero-based: both values shift
        {"2": "x"},  # single off-origin: "x" would render as {{1}}
    ):
        defect = build_parameters(variables)
        assert isinstance(defect, str), variables
        assert "1..N" in defect


def test_named_keys_become_named_parameters() -> None:
    """Named keys become named parameters."""
    params = build_parameters({"customer_name": "Priya"})
    assert params == [
        {"type": "text", "parameter_name": "customer_name", "text": "Priya"}
    ]


def test_no_variables_means_no_components() -> None:
    """No variables means no components."""
    assert build_parameters({}) == []


def test_mixed_key_styles_are_refused_not_guessed() -> None:
    """Mixed key styles are refused not guessed."""
    # Meta takes positional OR named per request, never both. The old
    # behaviour guessed named — emitting parameter_name='1' — and spent a
    # network round trip to receive the refusal this defect already states.
    defect = build_parameters({"1": "x", "otp": "y"})
    assert isinstance(defect, str)
    assert "mixes" in defect


def test_untextable_values_are_refused_not_coerced() -> None:
    """Untextable values are refused not coerced."""
    # str() rendered a JSON null as the literal word 'None' inside the
    # customer's message — corruption that LOOKS delivered. Numbers keep
    # their one obvious text form; everything else is a producer bug this
    # refusal surfaces.
    ok = build_parameters({"1": "Priya", "2": 42, "3": 9.5})
    assert ok == [
        {"type": "text", "text": "Priya"},
        {"type": "text", "text": "42"},
        {"type": "text", "text": "9.5"},
    ]
    for bad in (None, True, ["a"], {"a": 1}):
        defect = build_parameters({"1": "Priya", "2": bad})
        assert isinstance(defect, str), bad
        assert "'2'" in defect


def test_a_variable_defect_names_the_key_and_type_never_the_value() -> None:
    """A variable defect names the key and type never the value."""
    # Variable values can be personal data, and the defect string is
    # destined for a log line.
    defect = build_parameters({"otp": ["123456"]})
    assert isinstance(defect, str)
    assert "otp" in defect and "list" in defect
    assert "123456" not in defect


def test_a_unicode_digit_key_is_a_name_not_a_crash() -> None:
    """A unicode digit key is a name not a crash."""
    # '²'.isdigit() is True but int('²') raises: sorting by int() turned this
    # legal jsonb key into a mid-send exception that burned every attempt as
    # 'send_error'. As a (doomed) NAME, Meta's refusal is a classified,
    # terminal answer instead.
    assert build_parameters({"²": "x"}) == [
        {"type": "text", "parameter_name": "²", "text": "x"}
    ]


async def test_mixed_variables_are_blocked_before_posting(monkeypatch) -> None:
    """Mixed variables are blocked before posting — OUR refusal, not Meta's."""
    seen = _mocked(monkeypatch, _responds(200, ACCEPTED_BODY))
    outcome = await MetaWhatsAppAdapter().deliver(
        _message(variables={"1": "x", "otp": "y"}), _bundle(), _binding()
    )
    assert outcome.status == "blocked"
    assert outcome.reason == "template_variables_invalid"
    assert outcome.retryable is False
    # Nothing was posted: no rendering of a mixed dict is the right one.
    assert seen == {}


async def test_a_null_variable_is_blocked_before_posting(monkeypatch) -> None:
    """A null variable is blocked before posting — OUR refusal, not Meta's."""
    seen = _mocked(monkeypatch, _responds(200, ACCEPTED_BODY))
    outcome = await MetaWhatsAppAdapter().deliver(
        _message(variables={"1": "Priya", "2": None}), _bundle(), _binding()
    )
    assert outcome.status == "blocked"
    assert outcome.reason == "template_variables_invalid"
    assert outcome.retryable is False
    # Nothing was posted: 'Hi Priya, your order None…' must never exist.
    assert seen == {}


def test_the_language_comes_from_the_binding(monkeypatch) -> None:
    """The language comes from the binding."""
    # A per-endpoint fact: one merchant's templates may be approved in a
    # different locale from another's.
    adapter = MetaWhatsAppAdapter()
    parameters = build_parameters(_message().variables)
    assert isinstance(parameters, list)
    payload = adapter.build_payload(
        _message(),
        "919876543210",
        _binding(capabilities={"template_language": "hi"}),
        parameters,
    )
    assert payload["template"]["language"]["code"] == "hi"
    default = adapter.build_payload(_message(), "919876543210", _binding(), parameters)
    assert default["template"]["language"]["code"] == "en_US"


# --- refusals that never reach the network ----------------------------------


async def test_a_bundle_without_a_token_is_blocked(monkeypatch) -> None:
    """A missing bundle key is OUR refusal — 'blocked', the same status this
    reason carries from resolve_send_route, never Meta's word 'failed'."""
    seen = _mocked(monkeypatch, _responds(200, ACCEPTED_BODY))
    outcome = await MetaWhatsAppAdapter().deliver(
        _message(), CredentialBundle(values={"app_secret": "x"}), _binding()
    )
    assert outcome.status == "blocked"
    assert outcome.reason == "connector_credential_missing"
    assert outcome.retryable is False
    # Nothing was posted: a bundle missing its key cannot be fixed by asking
    # Meta about it.
    assert seen == {}


async def test_a_message_without_a_template_is_blocked(monkeypatch) -> None:
    """A message without a template is blocked — terminally, before posting."""
    seen = _mocked(monkeypatch, _responds(200, ACCEPTED_BODY))
    outcome = await MetaWhatsAppAdapter().deliver(
        _message(template_id=None), _bundle(), _binding()
    )
    assert outcome.status == "blocked"
    assert outcome.reason == "template_missing"
    assert outcome.retryable is False
    assert seen == {}


@pytest.mark.parametrize(
    "address", ["", "+1234", "not-a-number", "+" + "9" * 20, "+0123456789"]
)
async def test_an_unusable_address_is_blocked_before_posting(
    monkeypatch, address
) -> None:
    """An unusable address is blocked before posting — WE refused, Meta never
    saw it, so the manifest must not show the word reserved for Meta's no."""
    seen = _mocked(monkeypatch, _responds(200, ACCEPTED_BODY))
    outcome = await MetaWhatsAppAdapter().deliver(
        _message(sent_to_address=address), _bundle(), _binding()
    )
    assert outcome.status == "blocked"
    assert outcome.reason == "recipient_address_invalid"
    assert seen == {}


def test_recipient_normalisation_accepts_only_plausible_numbers() -> None:
    """Recipient normalisation accepts only plausible numbers."""
    # An Indian mobile is 12 digits in E.164: +91 plus the 10 national ones.
    assert to_meta_recipient("+91 98765-43210") == "919876543210"
    assert to_meta_recipient("9" * 16) is None  # past E.164's 15-digit ceiling
    assert to_meta_recipient("+12345") is None  # 5 digits, below any country
    assert to_meta_recipient("") is None


def test_the_accepted_length_window_matches_what_this_system_stores() -> None:
    """The accepted length window matches what this system stores."""
    # normalize.py and the platform_identity CHECK both allow +[1-9][0-9]{6,14}
    # — 7 to 15 digits. A tighter bound here would reject a number the system
    # was happy to store, and report it as an invalid address rather than as
    # the mismatch it is.
    from app.crm.shared.normalize import _E164

    for length in range(4, 18):
        stored = "+" + "9" * length
        assert (_E164.match(stored) is not None) == (
            to_meta_recipient(stored) is not None
        ), length
    # The [1-9] half of the same parity: no country code starts with 0, and
    # normalize.py refuses to store one — so accepting it here would post a
    # number the system would never have stored, and report Meta's code
    # instead of our recipient_address_invalid.
    assert _E164.match("+0123456789") is None
    assert to_meta_recipient("+0123456789") is None
    assert to_meta_recipient("0123456789") is None


# --- reading Meta's answer ---------------------------------------------------


async def test_an_accepted_send_records_the_wamid(monkeypatch) -> None:
    """An accepted send records the wamid."""
    outcome, _ = await _deliver(monkeypatch, _responds(200, ACCEPTED_BODY))
    assert outcome.status == "accepted"
    assert outcome.provider_message_id == "wamid.HBgMOTE5ODc2NTQzMjEw"


async def test_a_2xx_without_a_wamid_is_still_accepted(monkeypatch) -> None:
    """A 2xx without a wamid is still accepted."""
    # Meta took it. Calling this a failure would retry a message the customer
    # may already have — losing the receipt link is the smaller harm.
    outcome, _ = await _deliver(monkeypatch, _responds(200, {"messages": []}))
    assert outcome.status == "accepted"
    assert outcome.provider_message_id is None


@pytest.mark.parametrize("code", sorted(RETRYABLE_CODES))
async def test_pacing_errors_are_retryable(monkeypatch, code) -> None:
    """Pacing errors are retryable."""
    outcome, _ = await _deliver(
        monkeypatch, _responds(400, {"error": {"code": int(code), "message": "slow"}})
    )
    assert outcome.status == "failed"
    assert outcome.reason == code
    assert outcome.retryable is True
    # Pacing is not a verdict on the connection.


@pytest.mark.parametrize("code", sorted(TERMINAL_CODES))
async def test_message_level_refusals_never_retry(monkeypatch, code) -> None:
    """Message level refusals never retry."""
    outcome, _ = await _deliver(
        monkeypatch, _responds(400, {"error": {"code": int(code), "message": "no"}})
    )
    assert outcome.status == "failed"
    # The provider's own code, verbatim: a merchant asking "why" gets an
    # answer that matches Meta's documentation.
    assert outcome.reason == code
    assert outcome.retryable is False


@pytest.mark.parametrize("code", sorted(CREDENTIAL_CODES))
async def test_credential_refusals_flag_the_connection(monkeypatch, code) -> None:
    """Credential refusals flag the connection."""
    outcome, _ = await _deliver(
        monkeypatch,
        _responds(401, {"error": {"code": int(code), "message": "bad token"}}),
    )
    assert outcome.status == "failed"
    assert outcome.retryable is False
    # The provider's code lands on the row verbatim. That IS the signal the
    # channel module watches to decide the connection needs re-authenticating
    # — the send path deliberately does not act on it itself.
    assert outcome.reason == code


async def test_a_429_without_a_code_is_still_retryable(monkeypatch) -> None:
    """A 429 without a code is still retryable."""
    outcome, _ = await _deliver(monkeypatch, _responds(429, {}))
    assert outcome.retryable is True


async def test_an_unknown_5xx_is_retryable_and_an_unknown_4xx_is_not(
    monkeypatch,
) -> None:
    """An unknown 5xx is retryable and an unknown 4xx is not."""
    # Meta's problem may pass; ours will not, and three attempts would learn
    # nothing.
    server, _ = await _deliver(monkeypatch, _responds(503, {}))
    assert server.retryable is True
    assert server.reason == "http_503"

    client, _ = await _deliver(
        monkeypatch, _responds(400, {"error": {"code": 999999, "message": "?"}})
    )
    assert client.retryable is False
    assert client.reason == "999999"


async def test_a_provider_error_echoing_the_recipient_never_reaches_the_log(
    monkeypatch,
) -> None:
    """A provider error echoing the recipient never reaches the log."""
    # Meta's catalog strings carry no values today. This pins that even if
    # that contract breaks — or a proxy rewrites the body — the echoed
    # number dies at the log boundary, while the code survives for
    # classification and support.
    lines = []

    class _Recorder:
        def warning(self, msg):
            """Collect the log line."""
            lines.append(msg)

        def error(self, msg):
            """Collect the log line."""
            lines.append(msg)

        def info(self, msg):
            """Collect the log line."""
            lines.append(msg)

    monkeypatch.setattr(whatsapp_module, "logger", _Recorder())
    outcome, _ = await _deliver(
        monkeypatch,
        _responds(
            400,
            {"error": {"code": 100, "message": "Invalid parameter: to=919876543210"}},
        ),
    )
    assert outcome.reason == "100"
    joined = " ".join(lines)
    assert "919876543210" not in joined
    assert "code=100" in joined


async def test_a_non_json_response_does_not_crash_the_worker(monkeypatch) -> None:
    """A non json response does not crash the worker."""
    # A load balancer returning HTML must degrade to "failed, no detail",
    # not raise a JSONDecodeError that reads like a code bug.
    outcome, _ = await _deliver(
        monkeypatch, _responds(502, text="<html>bad gateway</html>")
    )
    assert outcome.status == "failed"
    assert outcome.retryable is True


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.PoolTimeout("x")],
)
async def test_a_transport_failure_is_retryable(monkeypatch, error) -> None:
    """A transport failure is retryable."""

    # "No answer" is not "no": the provider may have taken it.
    def handler(request: httpx.Request) -> httpx.Response:
        """Test double: canned provider response."""
        raise error

    outcome, _ = await _deliver(monkeypatch, handler)
    assert outcome.status == "failed"
    assert outcome.reason == "transport_error"
    assert outcome.retryable is True


# --- the classification table itself ----------------------------------------


def test_no_error_code_is_claimed_by_two_classes() -> None:
    """No error code is claimed by two classes."""
    # An overlap would make the outcome depend on the order of the branches
    # in read_response, which is exactly the kind of bug that shows up as
    # "sometimes it retries".
    assert RETRYABLE_CODES & TERMINAL_CODES == set()
    assert RETRYABLE_CODES & CREDENTIAL_CODES == set()
    assert TERMINAL_CODES & CREDENTIAL_CODES == set()


def test_the_endpoint_is_built_from_the_configured_dials() -> None:
    """The endpoint is built from the configured dials."""
    adapter = MetaWhatsAppAdapter(
        base_url="http://localhost:9999/", api_version="v99.0"
    )
    assert adapter.endpoint("PN1") == "http://localhost:9999/v99.0/PN1/messages"


def test_a_malformed_address_cannot_become_url_structure() -> None:
    """A malformed address cannot become url structure."""
    # The address column has no format CHECK and no writer validates it. A
    # '/' or '?' in a bad row must stay inside its one path segment — the
    # alternative posts the merchant's bearer token to whatever Graph path
    # the junk spells out.
    adapter = MetaWhatsAppAdapter(base_url="http://stub", api_version="v23.0")
    assert (
        adapter.endpoint("123/other?x=")
        == "http://stub/v23.0/123%2Fother%3Fx%3D/messages"
    )


async def test_a_control_character_address_does_not_escape_deliver(
    monkeypatch,
) -> None:
    """A control character address does not escape deliver."""
    # Unquoted, a '\n' in the address raised httpx.InvalidURL — which is not
    # an httpx.HTTPError — straight past the transport catch, and the row
    # burned every attempt as 'send_error'. Quoted, the request is made and
    # Meta's refusal comes back as a classified, terminal outcome.
    outcome, seen = await _deliver(
        monkeypatch,
        _responds(400, {"error": {"code": 100, "message": "no"}}),
        binding=_binding(address="PN\n1"),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "100"
    assert outcome.retryable is False
    assert "/PN%0A1/messages" in seen["url"]


# ===========================================================================
# Inbound: Meta's callback shape
#
# The door itself is tested in test_webhooks.py against a fake provider. What
# is tested here is only what is Meta's — how they sign, how they challenge,
# and how their envelope nests — because a second provider will do all three
# differently.
# ===========================================================================

APP_SECRET = "meta-app-secret-for-tests"
VERIFY_TOKEN = "meta-verify-token-for-tests"
PHONE_NUMBER_ID = "812345678901234"
OUT_WAMID = "wamid.OUTBOUND"
IN_WAMID = "wamid.INBOUND"
# 2026-08-31 12:00:00 UTC, as Meta sends it: unix seconds, as a string.
WA_TS = "1788177600"


@pytest.fixture
def meta_secrets(monkeypatch):
    """Both platform secrets present — the normal running state."""
    monkeypatch.setattr(whatsapp_module, "META_APP_SECRET", APP_SECRET)
    monkeypatch.setattr(whatsapp_module, "META_WEBHOOK_VERIFY_TOKEN", VERIFY_TOKEN)


def _wa_signature(raw: bytes, secret: str = APP_SECRET) -> str:
    """Sign the body the way Meta does."""
    import hashlib
    import hmac

    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _wa_status(**overrides) -> dict:
    """A Meta status object for notification payloads."""
    fields = {"id": OUT_WAMID, "status": "delivered", "timestamp": WA_TS}
    fields.update(overrides)
    return fields


def _wa_message(**overrides) -> dict:
    """A Meta inbound message object for notification payloads."""
    fields = {
        "from": "919876543210",
        "id": IN_WAMID,
        "timestamp": WA_TS,
        "type": "text",
        "text": {"body": "yes, confirm it"},
    }
    fields.update(overrides)
    return fields


def _wa_value(**overrides) -> dict:
    """A Meta change-value envelope for notification payloads."""
    fields = {"metadata": {"phone_number_id": PHONE_NUMBER_ID}}
    fields.update(overrides)
    return fields


def _wa_body(*values) -> dict:
    """A full Meta notification body around the given values."""
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "waba-1", "changes": [{"value": v} for v in values]}],
    }


# --- their signature ---------------------------------------------------------


def test_a_correct_signature_over_these_exact_bytes_passes(meta_secrets) -> None:
    """A correct signature over these exact bytes passes."""
    raw = b'{"entry":[]}'
    adapter = MetaWhatsAppAdapter()
    assert verify_signature(raw, {"X-Hub-Signature-256": _wa_signature(raw)})


def test_the_signature_header_is_matched_whatever_its_casing(meta_secrets) -> None:
    # Header names are case-insensitive on the wire, and a plain dict is not.
    """The signature header is matched whatever its casing."""
    raw = b'{"entry":[]}'
    for header in ("x-hub-signature-256", "X-HUB-SIGNATURE-256"):
        assert verify_signature(raw, {header: _wa_signature(raw)})


def test_a_signature_for_different_bytes_is_refused(meta_secrets) -> None:
    # The whole point: the MAC covers the body, so a real signature cannot be
    # replayed onto a payload someone else wrote.
    """A signature for different bytes is refused."""
    raw = b'{"entry":[]}'
    assert not verify_signature(
        b'{"entry":[{"tampered":true}]}',
        {"X-Hub-Signature-256": _wa_signature(raw)},
    )


def test_a_signature_from_a_different_secret_is_refused(meta_secrets) -> None:
    """A signature from a different secret is refused."""
    raw = b'{"entry":[]}'
    assert not verify_signature(
        raw, {"X-Hub-Signature-256": _wa_signature(raw, "not-our-secret")}
    )


@pytest.mark.parametrize(
    "header", [None, "", "deadbeef", "sha1=deadbeef", "sha256=", "sha256=not-hex"]
)
def test_a_missing_or_misshapen_signature_is_refused(meta_secrets, header) -> None:
    """A missing or misshapen signature is refused."""
    headers = {} if header is None else {"X-Hub-Signature-256": header}
    assert not verify_signature(b'{"entry":[]}', headers)


def test_without_a_configured_secret_everything_is_refused(monkeypatch) -> None:
    # Fail closed, and the loudest case in the file: an unauthenticated route
    # with no way to verify anything must refuse ALL traffic, never accept it.
    """Without a configured secret everything is refused."""
    monkeypatch.setattr(whatsapp_module, "META_APP_SECRET", "")
    raw = b'{"entry":[]}'
    assert not verify_signature(raw, {"X-Hub-Signature-256": _wa_signature(raw)})


# --- their handshake ---------------------------------------------------------


def test_the_handshake_echoes_the_challenge_for_our_token(meta_secrets) -> None:
    """The handshake echoes the challenge for our token."""
    assert (
        handshake_challenge(
            {
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "12345",
            }
        )
        == "12345"
    )


@pytest.mark.parametrize(
    "params",
    [
        {"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "1"},
        {
            "hub.mode": "unsubscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1",
        },
        {"hub.verify_token": VERIFY_TOKEN, "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN},
        {},
    ],
)
def test_the_handshake_refuses_anything_else(meta_secrets, params) -> None:
    """The handshake refuses anything else."""
    assert handshake_challenge(params) is None


def test_without_a_configured_token_no_handshake_succeeds(monkeypatch) -> None:
    """Without a configured token no handshake succeeds."""
    monkeypatch.setattr(whatsapp_module, "META_WEBHOOK_VERIFY_TOKEN", "")
    assert (
        handshake_challenge(
            {"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"}
        )
        is None
    )


# --- their envelope ----------------------------------------------------------


def test_a_status_becomes_one_letter_per_transition() -> None:
    # Meta sends one webhook per transition on the SAME message id, so the id
    # alone would collapse four letters into one and the journey would show
    # only whichever arrived last.
    """A status becomes one letter per transition."""
    letters = read_notification(_wa_body(_wa_value(statuses=[_wa_status()])))
    assert len(letters) == 1
    assert letters[0].topic == "message.status"
    assert letters[0].external_id == f"{OUT_WAMID}:delivered"
    assert letters[0].address == PHONE_NUMBER_ID
    assert letters[0].occurred_at == datetime.fromtimestamp(int(WA_TS), tz=timezone.utc)


def test_each_transition_of_one_message_files_separately() -> None:
    """Each transition of one message files separately."""
    ids = [
        read_notification(_wa_body(_wa_value(statuses=[_wa_status(status=s)])))[
            0
        ].external_id
        for s in ("sent", "delivered", "read")
    ]
    assert ids == [
        f"{OUT_WAMID}:sent",
        f"{OUT_WAMID}:delivered",
        f"{OUT_WAMID}:read",
    ]


def test_an_inbound_message_is_keyed_by_its_own_id() -> None:
    """An inbound message is keyed by its own id."""
    letters = read_notification(_wa_body(_wa_value(messages=[_wa_message()])))
    assert letters[0].topic == "message.inbound"
    assert letters[0].external_id == IN_WAMID
    assert letters[0].payload["text"] == {"body": "yes, confirm it"}


def test_both_kinds_ride_one_notification() -> None:
    # Receipts and replies routinely share an envelope; dropping either half
    # would lose real events with no error anywhere.
    """Both kinds ride one notification."""
    letters = read_notification(
        _wa_body(_wa_value(statuses=[_wa_status()], messages=[_wa_message()]))
    )
    assert [letter.topic for letter in letters] == [
        "message.status",
        "message.inbound",
    ]


def test_every_letter_carries_the_receiving_number() -> None:
    # The item itself never names which of OUR numbers it arrived on, and a
    # consumer needs that to find the binding.
    """Every letter carries the receiving number."""
    letters = read_notification(
        _wa_body(_wa_value(statuses=[_wa_status()], messages=[_wa_message()]))
    )
    for letter in letters:
        assert letter.payload["metadata"] == {"phone_number_id": PHONE_NUMBER_ID}


def test_the_payload_is_metas_object_not_our_reading_of_it() -> None:
    # Canon: raw, always. A consumer must see what Meta sent, including the
    # fields this module has no opinion about.
    """The payload is metas object not our reading of it."""
    raw = _wa_status(pricing={"billable": True, "category": "utility"}, extra="kept")
    letter = read_notification(_wa_body(_wa_value(statuses=[raw])))[0]
    assert letter.payload["pricing"] == {"billable": True, "category": "utility"}
    assert letter.payload["extra"] == "kept"


def test_a_reply_keeps_the_message_it_quotes() -> None:
    # context.id is the "replied to" link a consumer joins on.
    """A reply keeps the message it quotes."""
    letter = read_notification(
        _wa_body(_wa_value(messages=[_wa_message(context={"id": OUT_WAMID})]))
    )[0]
    assert letter.payload["context"]["id"] == OUT_WAMID


def test_several_entries_and_changes_are_all_found() -> None:
    # Meta batches freely, and they may concern different merchants' numbers.
    """Several entries and changes are all found."""
    body = {
        "entry": [
            {
                "changes": [
                    {"value": _wa_value(statuses=[_wa_status()])},
                    {"value": _wa_value(messages=[_wa_message()])},
                ]
            },
            {
                "changes": [
                    {
                        "value": _wa_value(
                            metadata={"phone_number_id": "999"},
                            statuses=[_wa_status(id="w2")],
                        )
                    }
                ]
            },
        ]
    }
    letters = read_notification(body)
    assert len(letters) == 3
    assert {letter.address for letter in letters} == {PHONE_NUMBER_ID, "999"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"entry": None},
        {"entry": "nope"},
        {"entry": [None, "x", 42]},
        {"entry": [{"changes": None}]},
        {"entry": [{"changes": [None, {"value": "not a dict"}]}]},
        # A value with no metadata names no endpoint to file it under —
        # template-status notifications arrive on this same webhook.
        {
            "entry": [
                {"changes": [{"value": {"statuses": [{"id": "x", "status": "y"}]}}]}
            ]
        },
    ],
)
def test_an_unreadable_envelope_yields_nothing_and_raises_nothing(body) -> None:
    """An unreadable envelope yields nothing and raises nothing."""
    assert read_notification(body) == []


def test_a_malformed_entry_does_not_hide_a_good_one() -> None:
    """A malformed entry does not hide a good one."""
    body = {
        "entry": [None, {"changes": [{"value": _wa_value(statuses=[_wa_status()])}]}]
    }
    assert len(read_notification(body)) == 1


@pytest.mark.parametrize(
    "value",
    [
        {"statuses": [{"id": OUT_WAMID}]},  # no status
        {"statuses": [{"status": "read"}]},  # no id
        {"statuses": ["not a dict"]},
        {"messages": [{"from": "91"}]},  # no id
        {"messages": [None]},
        {"statuses": "nope", "messages": 42},
    ],
)
def test_unusable_items_are_skipped_rather_than_filed(value) -> None:
    """Unusable items are skipped rather than filed."""
    body = _wa_body(_wa_value(**value))
    assert read_notification(body) == []


@pytest.mark.parametrize("bad", [None, "", "not-a-number", {}, [], "9" * 30])
def test_an_unusable_timestamp_never_raises(bad) -> None:
    # A letter with a broken clock is still worth filing.
    """An unusable timestamp never raises."""
    letter = read_notification(
        _wa_body(_wa_value(statuses=[_wa_status(timestamp=bad)]))
    )[0]
    assert letter.occurred_at is None
