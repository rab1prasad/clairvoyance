"""Turning a connected account's webhooks on at its provider's end.

Registering a callback URL routes NOTHING by itself — each merchant's account
must be subscribed separately. Most tests here are about the ways that can
fail, because a subscription that silently failed looks exactly like a healthy
account until somebody wonders why no events ever arrive.

Two layers, tested apart: subscribe.py finds the account, proves it is this
merchant's and fetches its credentials, and providers/whatsapp.py makes Meta's
actual call (tested against a mocked transport).
"""

from typing import Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.crm.connectivity import subscribe as subscribe_module
from app.crm.connectivity.providers import whatsapp as whatsapp_module
from app.crm.connectivity.providers.whatsapp import (
    WebhookSubscriptionError,
    subscribe_account,
    subscribe_to_webhooks,
)
from app.crm.connectivity.schemas import ConnectorInstallation, CredentialBundle
from app.crm.connectivity.subscribe import (
    SubscriptionRefused,
    account_id_of,
    subscribe_installation,
)
from app.schemas import Credential, CredentialType

WABA_ID = "waba-123"
TOKEN = "system-user-token"


def _installation(**overrides) -> ConnectorInstallation:
    """A healthy connected account; overrides replace any field."""
    fields = dict(
        id="i-1",
        merchant_id="shop",
        connector_key="whatsapp",
        external_account_id=WABA_ID,
        credential_id="cred-1",
        status="healthy",
    )
    fields.update(overrides)
    return ConnectorInstallation(**fields)


def _bundle(**values) -> CredentialBundle:
    """A bundle holding a usable WhatsApp token."""
    return CredentialBundle(values={"system_user_token": TOKEN, **values})


def _credential(**overrides) -> Credential:
    """A live vault row as get_credential_by_id(mask=False) returns it."""
    fields = dict(
        id="cred-1",
        reseller_id=None,
        name="wa-bundle",
        credential_type=CredentialType.CUSTOM,
        value={"system_user_token": TOKEN},
        is_active=True,
    )
    fields.update(overrides)
    return Credential(**fields)


class _FakeAccessor:
    """Stands in for db/accessor: one installation. The vault is NOT here —
    `credentials` belongs to app/database and is read through its own
    accessor, so it is stubbed separately."""

    def __init__(self, installation=None):
        """Test double."""
        self._installation = installation
        self.lookups: list = []

    async def get_installation(self, merchant_id, installation_id):
        """Test double: the seeded installation."""
        self.lookups.append((merchant_id, installation_id))
        return self._installation


class _Recorder:
    """Stands in for the provider call: records it, or refuses like Meta."""

    def __init__(self, refuse_with: Optional[str] = None):
        """Test double."""
        self.calls: list = []
        self._refuse_with = refuse_with

    async def __call__(self, account_id, bundle) -> None:
        """Test double: scripted behaviour."""
        if self._refuse_with:
            raise WebhookSubscriptionError(self._refuse_with)
        self.calls.append((account_id, bundle))


@pytest.fixture
def adapter(monkeypatch) -> _Recorder:
    """The Meta call, replaced by a recorder."""
    recording = _Recorder()
    monkeypatch.setattr(subscribe_module, "subscribe_account", recording)
    return recording


def _patch_vault(monkeypatch, credential) -> list:
    """The vault read, without a database. Returns the calls it received."""
    seen: list = []

    async def _get(credential_id, mask=True):
        """Test double: the seeded vault credential."""
        # Pinned: Meta needs the real token, never the API's mask.
        assert mask is False
        seen.append(credential_id)
        return credential

    monkeypatch.setattr(subscribe_module, "get_credential_by_id", _get)
    return seen


async def _refusal(monkeypatch, accessor, credential=None) -> str:
    """Run subscribe_installation expecting a refusal; return its sentence."""
    monkeypatch.setattr(subscribe_module, "accessor", accessor)
    _patch_vault(monkeypatch, credential)
    with pytest.raises(SubscriptionRefused) as caught:
        await subscribe_installation("shop", "i-1")
    return str(caught.value)


# --- picking the account and the adapter (no provider knowledge here) --------


async def test_a_healthy_account_is_subscribed_with_its_own_credentials(
    monkeypatch, adapter
) -> None:
    """A healthy account is subscribed with its own credentials."""
    monkeypatch.setattr(
        subscribe_module, "accessor", _FakeAccessor(installation=_installation())
    )
    read = _patch_vault(monkeypatch, _credential())
    assert await subscribe_installation("shop", "i-1") == WABA_ID
    # The account's own id and its own credentials — never a platform-wide
    # token, and the bundle rather than a token, so the key's name stays in
    # Meta's file.
    account_id, bundle = adapter.calls[0]
    assert account_id == WABA_ID
    assert bundle.secret("system_user_token") == TOKEN
    # Fetched by the installation's own credential_id: the bundle follows the
    # merchant-scoped row, so it cannot be another tenant's.
    assert read == ["cred-1"]


def test_the_vault_is_read_through_its_own_accessor() -> None:
    # Table-ownership law: `credentials` belongs to app/database, so this
    # module reads it exactly where send.py does — through that layer's
    # accessor, never SQL of its own.
    """The vault is read through its own accessor."""
    assert (
        subscribe_module.get_credential_by_id.__module__
        == "app.database.accessor.breeze_buddy.credentials"
    )


async def test_an_installation_of_another_tenant_is_simply_not_found(
    monkeypatch, adapter
) -> None:
    # The merchant scope is applied in the lookup, not checked afterwards, so
    # there is no branch here that could be forgotten.
    """An installation of another tenant is simply not found."""
    accessor = _FakeAccessor(installation=None)
    reason = await _refusal(monkeypatch, accessor)
    assert "No such connected account" in reason
    assert accessor.lookups == [("shop", "i-1")]
    assert adapter.calls == []


async def test_a_connector_with_no_subscription_step_is_refused(
    monkeypatch, adapter
) -> None:
    # The else of the connector branch. Another connector has its own
    # subscription mechanics, or none at all; running Meta's against it would
    # send a request nothing there understands.
    """A connector with no subscription step is refused."""
    reason = await _refusal(
        monkeypatch,
        _FakeAccessor(installation=_installation(connector_key="carrier_pigeon")),
        _credential(),
    )
    assert "carrier_pigeon" in reason
    assert adapter.calls == []


async def test_an_unsupported_connector_costs_no_vault_read(
    monkeypatch, adapter
) -> None:
    # What branching FIRST buys: the connector is decided before anything is
    # gathered, so an account we cannot subscribe is refused by name instead
    # of spending a vault read and then reporting whichever gathering step
    # happened to fail first.
    """An unsupported connector costs no vault read."""
    monkeypatch.setattr(
        subscribe_module,
        "accessor",
        _FakeAccessor(installation=_installation(connector_key="carrier_pigeon")),
    )
    read = _patch_vault(monkeypatch, _credential())
    with pytest.raises(SubscriptionRefused):
        await subscribe_installation("shop", "i-1")
    assert read == []


async def test_an_account_without_a_provider_id_is_refused(
    monkeypatch, adapter
) -> None:
    """An account without a provider id is refused."""
    reason = await _refusal(
        monkeypatch,
        _FakeAccessor(installation=_installation(external_account_id="")),
        _credential(),
    )
    # Named in the provider's own words, because that is what the merchant
    # has to go and find in Meta's console.
    assert "WhatsApp Business Account id" in reason
    assert adapter.calls == []


def test_the_shared_gathering_is_connector_agnostic() -> None:
    # The two helpers a future branch reuses: neither hard-codes WhatsApp, so
    # adding a connector is adding a branch and not a second copy of these.
    """The shared gathering is connector agnostic."""
    installation = _installation(external_account_id="")
    with pytest.raises(SubscriptionRefused) as caught:
        account_id_of(installation, "Instagram Professional Account")
    assert "Instagram Professional Account id" in str(caught.value)
    assert account_id_of(_installation(), "anything") == WABA_ID


@pytest.mark.parametrize(
    "installation_kwargs,credential,expected",
    [
        # Never connected, or connected and since disconnected.
        (dict(credential_id=None), None, "no stored credentials"),
        # The vault row is gone, or the accessor folded a DB error into None.
        (dict(), None, "missing or unreadable"),
        # Deactivated: still there, deliberately not usable.
        (dict(), _credential(is_active=False), "missing or unreadable"),
        # It would not decrypt — an undecryptable value decodes as {}.
        (dict(), _credential(value={}), "missing or unreadable"),
    ],
)
async def test_an_account_without_usable_credentials_is_refused(
    monkeypatch, adapter, installation_kwargs, credential, expected
) -> None:
    # Fail closed: handing the adapter no credentials would spend a round trip
    # to be told what we already know.
    """An account without usable credentials is refused."""
    reason = await _refusal(
        monkeypatch,
        _FakeAccessor(installation=_installation(**installation_kwargs)),
        credential,
    )
    assert expected in reason
    assert adapter.calls == []


async def test_a_refusal_from_meta_becomes_a_refusal_here(monkeypatch) -> None:
    # Meta's own words reach the merchant, not our paraphrase.
    """A refusal from meta becomes a refusal here."""
    monkeypatch.setattr(
        subscribe_module,
        "subscribe_account",
        _Recorder(refuse_with="Meta said no (code=200)"),
    )
    reason = await _refusal(
        monkeypatch, _FakeAccessor(installation=_installation()), _credential()
    )
    assert "code=200" in reason


# --- the WhatsApp adapter's actual Graph call --------------------------------


def _meta_says(status: int, body: Optional[dict] = None):
    """A canned Meta response, capturing the request that reached it."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Test double: records the request, then answers."""
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["headers"] = dict(request.headers)
        return httpx.Response(status, json=body or {})

    return handler, seen


@pytest.fixture
def meta(monkeypatch):
    """Point the provider's HTTP client at a canned responder."""

    def _install(handler):
        """A seeded installation for the scenario."""
        monkeypatch.setattr(
            whatsapp_module,
            "create_http_client",
            lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    return _install


async def test_the_subscription_posts_to_the_accounts_subscribed_apps(meta) -> None:
    """The subscription posts to the accounts subscribed apps."""
    handler, seen = _meta_says(200, {"success": True})
    meta(handler)
    await subscribe_to_webhooks(WABA_ID, TOKEN)

    assert seen["method"] == "POST"
    # Per-account, not per-number: subscribing routes every number under this
    # WhatsApp Business Account to our callback URL.
    assert seen["url"].endswith(f"/{WABA_ID}/subscribed_apps")
    assert seen["headers"]["authorization"] == f"Bearer {TOKEN}"


async def test_the_provider_pulls_its_own_token_from_the_bundle(meta) -> None:
    # whatsapp.py knows which key Meta wants; subscribe.py must not.
    """The provider pulls its own token from the bundle."""
    handler, seen = _meta_says(200, {"success": True})
    meta(handler)
    await subscribe_account(WABA_ID, _bundle())
    assert seen["headers"]["authorization"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize(
    "bundle",
    [
        CredentialBundle(values={}),
        CredentialBundle(values={"app_secret": "x"}),
        CredentialBundle(values={"system_user_token": ""}),
    ],
)
async def test_a_bundle_without_the_token_is_refused(meta, bundle) -> None:
    """A bundle without the token is refused."""
    handler, seen = _meta_says(200, {"success": True})
    meta(handler)
    with pytest.raises(WebhookSubscriptionError) as caught:
        await subscribe_account(WABA_ID, bundle)
    assert "Reconnect it first." in str(caught.value)
    # Nothing was posted: a bundle missing its key cannot be fixed by asking
    # Meta about it.
    assert seen == {}


async def test_a_waba_id_cannot_become_url_structure(meta) -> None:
    # The id comes from a database column with no format check. A '/' in it
    # must stay inside its path segment rather than steering the request —
    # and the merchant's bearer token — to some other Graph path.
    """A waba id cannot become url structure."""
    handler, seen = _meta_says(200, {"success": True})
    meta(handler)
    await subscribe_to_webhooks("waba/other?x=", TOKEN)
    assert "/waba%2Fother%3Fx%3D/subscribed_apps" in seen["url"]


async def test_metas_refusal_is_raised_with_its_own_words(meta) -> None:
    """Metas refusal is raised with its own words."""
    handler, _ = _meta_says(
        400, {"error": {"code": 200, "message": "Permissions error"}}
    )
    meta(handler)
    with pytest.raises(WebhookSubscriptionError) as caught:
        await subscribe_to_webhooks(WABA_ID, TOKEN)
    # A merchant asking "why" gets an answer matching Meta's documentation.
    assert "200" in str(caught.value)
    assert "Permissions error" in str(caught.value)


async def test_an_unreadable_refusal_still_raises(meta) -> None:
    # A load balancer returning HTML must not be read as success.
    """An unreadable refusal still raises."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Test double: an upstream error page."""
        return httpx.Response(502, text="<html>bad gateway</html>")

    meta(handler)
    with pytest.raises(WebhookSubscriptionError):
        await subscribe_to_webhooks(WABA_ID, TOKEN)


# --- the route ----------------------------------------------------------------


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """The subscribe route with its auth dependency stubbed out — this file
    tests the refusal mapping; test_webhooks.py tests that the dependency is
    present at all."""
    from app.crm import auth as crm_auth
    from app.crm.connectivity import api as connectivity_api

    app = FastAPI()
    app.include_router(connectivity_api.router, prefix="/crm/connectivity")
    app.dependency_overrides[crm_auth.crm_admin_user] = lambda: None
    return TestClient(app)


def test_the_route_reports_a_refusal_as_409(client, monkeypatch) -> None:
    """The route reports a refusal as 409."""

    async def _refuses(merchant_id, installation_id):
        """Test double: every refusal path, collapsed to one."""
        raise SubscriptionRefused("This account has no stored credentials.")

    monkeypatch.setattr("app.crm.connectivity.api.subscribe_installation", _refuses)
    response = client.post(
        "/crm/connectivity/installations/i-1/subscribe", params={"merchant_id": "shop"}
    )
    # 409, not 400: the request was understood and deliberately declined, and
    # the message says which so the caller can act on it.
    assert response.status_code == 409
    assert "no stored credentials" in response.json()["detail"]


def test_the_route_echoes_what_was_subscribed(client, monkeypatch) -> None:
    """The route echoes what was subscribed."""

    async def _subscribes(merchant_id, installation_id):
        """Test double: a successful subscription."""
        return WABA_ID

    monkeypatch.setattr("app.crm.connectivity.api.subscribe_installation", _subscribes)
    response = client.post(
        "/crm/connectivity/installations/i-1/subscribe", params={"merchant_id": "shop"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "installation_id": "i-1",
        "external_account_id": WABA_ID,
        "subscribed": True,
    }


def test_the_route_requires_a_merchant_scope(client) -> None:
    # Tenancy law: merchant_id is required, never inferred.
    """The route requires a merchant scope."""
    assert (
        client.post("/crm/connectivity/installations/i-1/subscribe").status_code == 422
    )
