"""Turning a connected account's webhooks on at its provider's end.

Registering a callback URL in a provider's dashboard routes NOTHING by
itself: it says where to deliver, but each merchant's account must separately
be subscribed to our app before any of that merchant's events are sent. This
module is that second step.

The connector is chosen by an explicit branch per connector, with the refusal
in the else — so ADDING a connector that supports subscription is adding a
branch, and nothing else in this file moves. That is written this way round
on purpose: a `if connector != whatsapp: refuse` guard would make the next
provider start by DELETING a refusal, which is the shape that quietly gets
half-done.

What is deliberately not here is a general `subscribe(channel)`. The step
itself is not universal — Meta subscribes an ACCOUNT to an app, Twilio wants a
callback URL set on a number, and an email provider is configured once for our
whole account with no per-merchant step at all — so each branch calls its own
provider's function with whatever that provider's mechanics need. What every
branch does share is gathered by the two helpers below.

It exists as its own endpoint rather than a step inside onboarding because
onboarding is not built yet (PR #1038). When it lands, its connect sequence
should call providers.whatsapp.subscribe_to_webhooks directly — the signature
was matched to the stub it left behind — and this endpoint becomes the repair
tool for accounts connected before webhooks existed, or whose subscription
was dropped at the provider's end.
"""

from typing import Optional

from app.core.logger import logger
from app.crm.connectivity.db import accessor
from app.crm.connectivity.providers.whatsapp import (
    WebhookSubscriptionError,
    subscribe_account,
)
from app.crm.connectivity.schemas import ConnectorInstallation, CredentialBundle
from app.database.accessor.breeze_buddy.credentials import get_credential_by_id

CONNECTOR_WHATSAPP = "whatsapp"


class SubscriptionRefused(Exception):
    """This account cannot be subscribed, and why. Carries a merchant-safe
    sentence: it is shown to whoever pressed the button."""


def account_id_of(installation: ConnectorInstallation, provider_term: str) -> str:
    """The provider's own id for this account, or the refusal.

    Every subscription call needs one, so it is gathered here rather than
    re-checked in each branch. ``provider_term`` is what that provider calls
    it ("WhatsApp Business Account"), because the merchant reading the refusal
    has to find it in that provider's console, not ours.
    """
    if not installation.external_account_id:
        raise SubscriptionRefused(
            f"This account has no {provider_term} id to subscribe."
        )
    return installation.external_account_id


async def bundle_of(installation: ConnectorInstallation) -> CredentialBundle:
    """This account's own secrets, or the refusal.

    The whole bundle rather than a token: which key a provider wants stays in
    that provider's file. Fail closed — handing a provider no credentials
    would spend a round trip to be told what is already known here.
    """
    if not installation.credential_id:
        raise SubscriptionRefused(
            "This account has no stored credentials. Reconnect it first."
        )
    # The vault (`credentials`) belongs to app/database, so it is read through
    # ITS accessor, exactly as send.py does — never SQL from here
    # (table-ownership law). mask=False: the provider needs the real secret.
    credential = await get_credential_by_id(installation.credential_id, mask=False)
    if credential is None or not credential.is_active or not credential.value:
        # Vault row gone, deactivated, or it would not decrypt (an
        # undecryptable value decodes as {}). One fact from here: there is no
        # usable secret to subscribe with.
        raise SubscriptionRefused(
            "This account's credentials are missing or unreadable. "
            "Reconnect it first."
        )
    return CredentialBundle(values=credential.value)


async def subscribe_installation(
    merchant_id: str, installation_id: str
) -> Optional[str]:
    """Subscribe one connected account to our app. Returns the provider's
    account id.

    Raises SubscriptionRefused with a sentence a merchant can act on. Fails
    closed at every step — no installation, wrong tenant, a connector with no
    subscription step, no credential, or a refusal from the provider all mean
    "not subscribed", and the caller is told which. A subscription that
    silently failed is indistinguishable from a healthy account until somebody
    wonders why no events ever arrive.

    The merchant scope is applied in the lookup, not checked afterwards, so an
    installation belonging to another tenant is simply not found.
    """
    installation = await accessor.get_installation(merchant_id, installation_id)
    if installation is None:
        raise SubscriptionRefused("No such connected account for this merchant.")

    try:
        # ONE BRANCH PER CONNECTOR THAT HAS A SUBSCRIPTION STEP. To add one:
        # an elif on its connector_key, gather with the helpers above, call
        # its own provider function. Nothing outside this block changes.
        if installation.connector_key == CONNECTOR_WHATSAPP:
            waba_id = account_id_of(installation, "WhatsApp Business Account")
            bundle = await bundle_of(installation)
            await subscribe_account(waba_id, bundle)
        else:
            # Every other connector, including ones that will never have this
            # step: an email provider is configured once for our whole account
            # and has nothing per-merchant to turn on. Refusing beats guessing
            # — running Meta's call against it would send a request nothing
            # there understands.
            raise SubscriptionRefused(
                f"This account is a '{installation.connector_key}' connector, "
                f"which has no webhook subscription to turn on."
            )
    except WebhookSubscriptionError as e:
        # The provider's own refusal, passed through: a merchant asking "why"
        # gets an answer matching that provider's documentation rather than
        # our paraphrase. Caught around the whole block so a second branch
        # inherits it — a provider raising its own error type adds that type
        # here (and by then the class belongs somewhere both can see).
        logger.error(f"connectivity: webhook subscription refused — {e}")
        raise SubscriptionRefused(str(e))

    logger.info(
        f"connectivity: subscribed {installation.connector_key} webhooks for "
        f"installation {installation.id}"
    )
    return installation.external_account_id
