"""The webhook door: what it refuses, what it files, and what it never touches.

The division of labour, mirrored from the code: Meta's own wire shape — how
they sign, how they challenge, how their envelope nests — is tested in
test_whatsapp_adapter.py. What is tested here is the door's own half, which
is provider-free and would survive a second provider unchanged: resolve which
merchant owns the endpoint, file the letters, and answer.

Two promises are under test, and the second matters as much as the first.
The door is the only unauthenticated route in the service, so most tests are
about refusing something. And the door only FILES — it advances no message,
creates no customer, opens no conversation. That is asserted mechanically
(see _FakeAccessor) rather than promised in a docstring, because the whole
point of this PR's scope is that consumers do the rest.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.crm.connectivity import api as connectivity_api, webhooks as webhooks_module
from app.crm.connectivity.providers import whatsapp as whatsapp_module
from app.crm.connectivity.schemas import (
    TOPIC_INBOUND,
    TOPIC_STATUS,
    ChannelBinding,
    InboundLetter,
)
from app.crm.connectivity.webhooks import (
    OUTCOME_ACCEPTED,
    OUTCOME_BAD_SIGNATURE,
    OUTCOME_UNREADABLE,
    file_letters,
    ingest_whatsapp,
    whatsapp_handshake,
)

SECRET = "app-secret-for-tests"
TOKEN = "verify-token-for-tests"
OUR_NUMBER = "812345678901234"
OTHER_NUMBER = "812999999999999"
OUT_WAMID = "wamid.OUTBOUND"
IN_WAMID = "wamid.INBOUND"
# 2026-08-31 12:00:00 UTC, as Meta sends it: unix seconds, as a string.
TS = "1788177600"
AT = datetime.fromtimestamp(int(TS), tz=timezone.utc)


def _status(**overrides) -> dict:
    """A delivery receipt as Meta sends one."""
    fields = {"id": OUT_WAMID, "status": "delivered", "timestamp": TS}
    fields.update(overrides)
    return fields


def _message(**overrides) -> dict:
    """An inbound customer message as Meta sends one."""
    fields = {
        "from": "919876543210",
        "id": IN_WAMID,
        "timestamp": TS,
        "type": "text",
        "text": {"body": "yes, confirm it"},
    }
    fields.update(overrides)
    return fields


def _value(number=OUR_NUMBER, **overrides) -> dict:
    """One notification value, addressed to one of our numbers."""
    fields = {"metadata": {"phone_number_id": number}}
    fields.update(overrides)
    return fields


def _body(*values) -> bytes:
    """Meta's envelope around the given values, as raw bytes."""
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [{"id": "waba-1", "changes": [{"value": v} for v in values]}],
        }
    ).encode()


def _signature(raw: bytes, secret: str = SECRET) -> str:
    """A valid X-Hub-Signature-256 over these exact bytes."""
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _signed(raw: bytes) -> dict:
    """Headers Meta would send with this body."""
    return {"X-Hub-Signature-256": _signature(raw)}


def _binding(**overrides) -> ChannelBinding:
    """An active WhatsApp binding; overrides replace any field."""
    fields = dict(
        id="b-1",
        merchant_id="shop",
        channel="whatsapp",
        installation_id="i-1",
        address=OUR_NUMBER,
        capabilities={},
        is_primary=True,
        status="active",
    )
    fields.update(overrides)
    return ChannelBinding(**fields)


class _FakeAccessor:
    """Stands in for db/accessor — and polices this PR's scope.

    The door is allowed exactly one call: reading which merchant owns the
    receiving endpoint. Everything else is recorded into ``other_calls``
    instead of raising, so a test can assert WHAT the door reached for rather
    than decoding an AttributeError.
    """

    def __init__(self, bindings=None):
        """Test double."""
        self._bindings = bindings if bindings is not None else {OUR_NUMBER: _binding()}
        self.lookups: list = []
        self.other_calls: list = []

    async def get_binding_by_address(self, channel, address):
        """Test double: the seeded binding, by address."""
        self.lookups.append((channel, address))
        return self._bindings.get(address)

    def __getattr__(self, name):
        """Record any unexpected call instead of raising."""

        async def _record(*args, **kwargs):
            """Record the call for the test's assertion."""
            self.other_calls.append(name)
            return None

        return _record


@pytest.fixture
def configured(monkeypatch):
    """Both platform secrets present — the normal running state."""
    monkeypatch.setattr(whatsapp_module, "META_APP_SECRET", SECRET)
    monkeypatch.setattr(whatsapp_module, "META_WEBHOOK_VERIFY_TOKEN", TOKEN)


@pytest.fixture
def spine(monkeypatch) -> list:
    """Captures every letter filed, which is the door's entire output."""
    letters: list = []

    async def _record(**kwargs):
        """Record the call for the test's assertion."""
        letters.append(kwargs)
        return f"ev-{len(letters)}"

    monkeypatch.setattr(webhooks_module, "record_event", _record)
    return letters


@pytest.fixture
def door(monkeypatch, spine, configured) -> _FakeAccessor:
    """The door with a stubbed database and spine, secrets configured."""
    fake = _FakeAccessor()
    monkeypatch.setattr(webhooks_module, "accessor", fake)
    return fake


# --- verification gates everything --------------------------------------------


async def test_an_unsigned_body_is_refused_before_it_is_parsed(door, spine) -> None:
    """An unsigned body is refused before it is parsed."""
    outcome = await ingest_whatsapp(_body(_value(statuses=[_status()])), {})
    assert outcome == OUTCOME_BAD_SIGNATURE
    # Nothing was read, nothing was looked up, nothing was filed.
    assert spine == [] and door.lookups == []


async def test_a_tampered_body_is_refused(door, spine) -> None:
    # The attack this stops: a real signature, a rewritten payload.
    """A tampered body is refused."""
    real = _body(_value(statuses=[_status()]))
    tampered = _body(_value(statuses=[_status(status="read")]))
    assert await ingest_whatsapp(tampered, _signed(real)) == OUTCOME_BAD_SIGNATURE
    assert spine == []


async def test_a_signed_non_object_body_is_unreadable(door, spine) -> None:
    """A signed non object body is unreadable."""
    assert await ingest_whatsapp(b"[1,2,3]", _signed(b"[1,2,3]")) == OUTCOME_UNREADABLE
    assert spine == []


@pytest.mark.parametrize("raw", [b"", b"not json", b"null", b"\xff\xfe"])
async def test_a_body_that_is_not_json_never_raises(door, spine, raw) -> None:
    """A body that is not json never raises."""
    assert await ingest_whatsapp(raw, _signed(raw)) == OUTCOME_UNREADABLE
    assert spine == []


def test_the_handshake_is_answered_by_the_provider(configured) -> None:
    # The door delegates; Meta's hub.* rules are tested with Meta's code.
    """The handshake is answered by the provider."""
    assert (
        whatsapp_handshake(
            {
                "hub.mode": "subscribe",
                "hub.verify_token": TOKEN,
                "hub.challenge": "9876",
            }
        )
        == "9876"
    )
    assert (
        whatsapp_handshake(
            {"hub.mode": "subscribe", "hub.verify_token": "guess", "hub.challenge": "1"}
        )
        is None
    )


# --- filing, and nothing else -------------------------------------------------


async def test_letters_are_filed_under_whoever_owns_the_endpoint(door, spine) -> None:
    """Letters are filed under whoever owns the endpoint."""
    raw = _body(_value(statuses=[_status()]))
    assert await ingest_whatsapp(raw, _signed(raw)) == OUTCOME_ACCEPTED

    assert door.lookups == [("whatsapp", OUR_NUMBER)]
    assert len(spine) == 1
    letter = spine[0]
    assert letter["merchant_id"] == "shop"
    assert letter["source"] == "whatsapp"
    assert letter["topic"] == TOPIC_STATUS
    assert letter["external_id"] == f"{OUT_WAMID}:delivered"
    # Meta's clock, not ours: when it happened is their fact.
    assert letter["occurred_at"] == AT


async def test_the_door_stamps_no_customer(door, spine) -> None:
    # Attaching a customer means finding or creating one, which writes a CRM
    # table. A consumer does that later (ADR 0020); the door must not.
    """The door stamps no customer."""
    raw = _body(_value(messages=[_message()]))
    await ingest_whatsapp(raw, _signed(raw))
    assert spine[0]["customer_id"] is None


async def test_the_door_writes_no_crm_table(door, spine) -> None:
    # THE scope test. One read to find the owner, and nothing else — no
    # message update, no customer resolve, no conversation.
    """The door writes no crm table."""
    raw = _body(_value(statuses=[_status()], messages=[_message()]))
    await ingest_whatsapp(raw, _signed(raw))
    assert {letter["topic"] for letter in spine} == {TOPIC_STATUS, TOPIC_INBOUND}
    assert door.other_calls == []


async def test_one_callback_can_carry_two_merchants(
    monkeypatch, spine, configured
) -> None:
    # Meta batches across accounts, so tenancy is resolved per endpoint — not
    # once per request.
    """One callback can carry two merchants."""
    monkeypatch.setattr(
        webhooks_module,
        "accessor",
        _FakeAccessor(
            bindings={
                OUR_NUMBER: _binding(),
                OTHER_NUMBER: _binding(
                    id="b-2", merchant_id="other", address=OTHER_NUMBER
                ),
            }
        ),
    )
    raw = _body(
        _value(statuses=[_status()]),
        _value(number=OTHER_NUMBER, messages=[_message(id="wamid.OTHER")]),
    )
    await ingest_whatsapp(raw, _signed(raw))
    assert {letter["merchant_id"] for letter in spine} == {"shop", "other"}


async def test_an_endpoint_costs_one_lookup_however_many_letters(door, spine) -> None:
    # Grouping by address first: a batch of receipts for one number must not
    # become one identical query per receipt.
    """An endpoint costs one lookup however many letters."""
    raw = _body(
        _value(statuses=[_status(id=f"wamid.{n}") for n in range(5)]),
    )
    await ingest_whatsapp(raw, _signed(raw))
    assert len(spine) == 5
    assert door.lookups == [("whatsapp", OUR_NUMBER)]


async def test_an_unowned_endpoint_files_nothing_and_spares_the_others(
    door, spine
) -> None:
    # A number we do not own, or no longer own. Filing it under any merchant
    # would be a cross-tenant leak — and it must not cost the letters beside
    # it, which belong to someone else entirely.
    """An unowned endpoint files nothing and spares the others."""
    counts = await file_letters(
        "whatsapp",
        [
            InboundLetter(address="nobodys", topic=TOPIC_STATUS, external_id="a"),
            InboundLetter(address=OUR_NUMBER, topic=TOPIC_STATUS, external_id="b"),
        ],
    )
    assert counts == {"unknown_endpoint": 1, TOPIC_STATUS: 1}
    assert [letter["external_id"] for letter in spine] == ["b"]


async def test_a_letter_the_spine_refuses_is_not_an_error(monkeypatch, door) -> None:
    # record_event returns None for a duplicate Meta re-sent. Canon: dedupe
    # once, for everyone — the door implements none of its own and still
    # answers 200, because asking for a resend would not help.
    """A letter the spine refuses is not an error."""

    async def _refuses(**kwargs):
        """Test double: the spine refuses this letter."""
        return None

    monkeypatch.setattr(webhooks_module, "record_event", _refuses)
    raw = _body(_value(statuses=[_status()]))
    assert await ingest_whatsapp(raw, _signed(raw)) == OUTCOME_ACCEPTED


async def test_a_callback_carrying_nothing_is_still_accepted(door, spine) -> None:
    # Account-review and template-status notifications ride the same webhook.
    """A callback carrying nothing is still accepted."""
    raw = _body({"template_status": "APPROVED"})
    assert await ingest_whatsapp(raw, _signed(raw)) == OUTCOME_ACCEPTED
    assert spine == [] and door.lookups == []


# --- the routes ---------------------------------------------------------------


@pytest.fixture
def client(door) -> TestClient:
    """The connectivity router over a stubbed door."""
    app = FastAPI()
    app.include_router(connectivity_api.router, prefix="/crm/connectivity")
    return TestClient(app)


URL = "/crm/connectivity/webhooks/whatsapp"


def test_a_signed_body_answers_200(client, spine) -> None:
    """A signed body answers 200."""
    raw = _body(_value(statuses=[_status()]))
    assert client.post(URL, content=raw, headers=_signed(raw)).status_code == 200
    assert len(spine) == 1


def test_an_unsigned_body_is_forbidden_and_files_nothing(client, spine) -> None:
    """An unsigned body is forbidden and files nothing."""
    response = client.post(URL, content=_body(_value()))
    assert response.status_code == 403
    assert spine == []
    # No detail: a caller who cannot sign has not earned an explanation.
    assert response.content == b""


def test_a_signed_non_object_body_is_refused_at_the_door(client, spine) -> None:
    """A signed non object body is refused at the door."""
    raw = b"[1,2,3]"
    assert client.post(URL, content=raw, headers=_signed(raw)).status_code == 400
    assert spine == []


def test_an_oversized_body_is_refused_before_everything_else(
    client, spine, monkeypatch
) -> None:
    """An oversized body is refused before everything else.

    Even a VALID signature does not buy an unbounded read: the cap fires
    while streaming, before verification, because buffering an arbitrary
    body for an unauthenticated caller is the cost the route must not pay.
    """
    monkeypatch.setattr(connectivity_api, "MAX_WEBHOOK_BODY_BYTES", 64)
    raw = b"x" * 65
    response = client.post(URL, content=raw, headers=_signed(raw))
    assert response.status_code == 413
    assert spine == []


def test_the_handshake_route_echoes_in_plain_text(client) -> None:
    """The handshake route echoes in plain text."""
    response = client.get(
        URL,
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": TOKEN,
            "hub.challenge": "9876",
        },
    )
    assert response.status_code == 200
    # Echoed raw — Meta compares the body, so quotes or JSON would fail it.
    assert response.text == "9876"


def test_the_handshake_route_hides_itself_from_a_wrong_token(client) -> None:
    """The handshake route hides itself from a wrong token."""
    response = client.get(
        URL,
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "guess",
            "hub.challenge": "9876",
        },
    )
    # 404, not 403: answering differently for a wrong token than for a missing
    # one tells an unauthenticated caller they found something worth guessing.
    assert response.status_code == 404


def test_only_the_webhook_routes_are_unauthenticated() -> None:
    """The carve-out is those two routes and nothing else.

    Both directions matter: auth appearing on the webhook pair locks Meta out
    and silently stops every event, and auth going missing from subscribe
    would expose connector administration to anyone.
    """
    by_path: dict = {}
    for route in connectivity_api.router.routes:
        path = getattr(route, "path", "")
        dependant = getattr(route, "dependant", None)
        names = [
            d.call.__name__
            for d in (dependant.dependencies if dependant else [])
            if getattr(d, "call", None) is not None
        ]
        by_path.setdefault(path, []).extend(names)

    assert by_path["/webhooks/whatsapp"] == []
    assert "crm_admin_user" in by_path["/installations/{installation_id}/subscribe"]
