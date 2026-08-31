"""connectivity — the public surface.

The only file other modules and app/crm/worker_main.py may import.

This module owns everything between "we want to send something" and "the
provider took it": connector accounts, the endpoints under them, the message
table, send() and its dispatcher. It is channel-agnostic — WhatsApp,
Instagram and email are adapters behind send(), not packages.

claim_sends/dispatch_send are the module's two callables for the shared
drain-loop scaffold (design/worker-runtime.md): worker_main registers them
as the "dispatcher" role. queue_message() is how a producer (the walker's
send node first) proposes a send: one queued row, no verdict.

The webhook pair is the inbound direction, and it is deliberately NOT a
worker. ingest_whatsapp_webhook verifies a Meta callback and files its letters
in the event spine; whatsapp_handshake_challenge answers Meta's subscription
challenge. Both name WhatsApp because it is the only provider that sends us
callbacks today — see webhooks.py for why no channel seam exists yet.
Consumers of those letters — the ones that advance a message to delivered, or
create the customer who replied — are a separate concern and not on this
surface yet.

subscribe_installation turns a connected account's webhooks on at Meta's end.
send() stays off this surface so that nothing outside can reach a provider.
"""

from app.crm.connectivity.dispatch import claim_sends, dispatch_send
from app.crm.connectivity.queue import queue_message
from app.crm.connectivity.subscribe import SubscriptionRefused, subscribe_installation
from app.crm.connectivity.webhooks import (
    OUTCOME_ACCEPTED,
    OUTCOME_BAD_SIGNATURE,
    OUTCOME_UNREADABLE,
    ingest_whatsapp as ingest_whatsapp_webhook,
    whatsapp_handshake as whatsapp_handshake_challenge,
)

__all__ = [
    "claim_sends",
    "dispatch_send",
    "ingest_whatsapp_webhook",
    "queue_message",
    "whatsapp_handshake_challenge",
    # The vocabulary ingest_whatsapp_webhook answers in. Exported so the route
    # maps outcomes to status codes by name instead of by string literal — a
    # renamed outcome then breaks at import, not in production.
    "OUTCOME_ACCEPTED",
    "OUTCOME_BAD_SIGNATURE",
    "OUTCOME_UNREADABLE",
    "subscribe_installation",
    "SubscriptionRefused",
]
