"""The webhook door: verify a Meta callback, file its letters, answer 200.

Two halves, kept apart on purpose.

Provider-shaped work — is this signature real, what does this envelope mean,
what does their handshake want — lives in providers/whatsapp.py. None of it
is here, and none of it is behind a general interface either: see the note in
providers/base.py for why webhook handling is not part of ChannelAdapter.
The short version is that the variation runs per PROVIDER (one Meta app
serves WhatsApp, Instagram and Messenger through one callback and one
signature scheme) rather than per channel, so a per-channel seam would split
identical code while leaving the real axis unabstracted. A seam gets built
when a second provider makes its actual shape known.

Provider-free work is what remains in this file, and it is the same for
anyone: which merchant owns the endpoint a letter arrived on, and file the
letters in the event spine. That is the whole job. Everything a reader might
expect next — advancing a message to 'delivered', creating the customer who
replied, opening a conversation — belongs to consumers that read the spine
afterwards. None of it happens here, and this path writes no CRM table.

The split is the canon's, not a preference: every fact from outside enters
through crm_event_raw, and the front door verifies, stores the raw letter, and
returns. Storing before understanding is what makes a payload replayable after
a consumer bug, and what keeps a slow consumer from becoming a webhook retry
storm — Meta re-sends anything not answered within seconds.
"""

import json
from typing import Dict, List, Mapping, Optional

from app.core.logger import logger
from app.crm.connectivity.db import accessor
from app.crm.connectivity.providers import whatsapp
from app.crm.connectivity.schemas import InboundLetter
from app.crm.record.contracts import record_event

CHANNEL_WHATSAPP = "whatsapp"

# What the door reports back to its route, which turns it into a status code.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_BAD_SIGNATURE = "bad_signature"
OUTCOME_UNREADABLE = "unreadable"


def whatsapp_handshake(params: Mapping[str, str]) -> Optional[str]:
    """The value Meta's subscription challenge expects back, or None."""
    return whatsapp.handshake_challenge(params)


async def file_letters(channel: str, letters: List[InboundLetter]) -> Dict[str, int]:
    """File letters in the spine, under whoever owns the endpoint each names.

    The provider-free half of the door, and the part a second provider would
    reuse unchanged.

    Letters are grouped by address first because one callback may legitimately
    carry facts for several merchants' numbers, and each endpoint then costs
    one lookup rather than one per letter. Each address is independent: one
    that no merchant owns must not cost the letters beside it, which belong to
    someone else entirely.
    """
    counts: Dict[str, int] = {}

    def tally(outcome: str) -> None:
        """One more letter landed on this outcome."""
        counts[outcome] = counts.get(outcome, 0) + 1

    by_address: Dict[str, List[InboundLetter]] = {}
    for letter in letters:
        by_address.setdefault(letter.address, []).append(letter)

    for address, group in by_address.items():
        binding = await accessor.get_binding_by_address(channel, address)
        if binding is None:
            # An endpoint we do not own, or no longer own. Filing it under any
            # merchant would be a cross-tenant leak, so it stops here.
            logger.warning(
                f"connectivity: {channel} webhook for an endpoint no merchant owns"
            )
            for _ in group:
                tally("unknown_endpoint")
            continue

        for letter in group:
            event_id = await record_event(
                merchant_id=binding.merchant_id,
                source=channel,
                topic=letter.topic,
                external_id=letter.external_id,
                payload=letter.payload,
                occurred_at=letter.occurred_at,
                # Deliberately unstamped. Attaching a customer means finding or
                # creating one, which writes a CRM table — a consumer's job,
                # not the door's (ADR 0020).
                customer_id=None,
            )
            # None is a duplicate Meta re-sent, or a spine that refused it.
            # Both are already logged there, and neither changes our answer:
            # the caller gets 200 either way, because asking for the
            # notification again would not help.
            tally(letter.topic if event_id else "not_filed")

    return counts


async def ingest_whatsapp(raw_body: bytes, headers: Mapping[str, str]) -> str:
    """Verify a Meta callback and file what it carries. Returns an outcome word.

    The route turns that word into a status code; keeping the mapping there
    means this stays testable without HTTP.
    """
    if not whatsapp.verify_signature(raw_body, headers):
        logger.warning(
            "connectivity: rejected a whatsapp webhook with an invalid signature"
        )
        return OUTCOME_BAD_SIGNATURE

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        body = None
    if not isinstance(body, dict):
        # Signed by Meta, but not shaped like anything they document. Worth a
        # 400 and a log line rather than a silent 200.
        logger.error("connectivity: a signed whatsapp body was not a JSON object")
        return OUTCOME_UNREADABLE

    counts = await file_letters(CHANNEL_WHATSAPP, whatsapp.read_notification(body))
    logger.info(f"connectivity: whatsapp webhook filed {counts or 'nothing'}")
    return OUTCOME_ACCEPTED
