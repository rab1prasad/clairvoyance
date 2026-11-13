"""Reporting webhooks are signed, or they are not sent.

``send_webhook_with_retry`` posts the lead's name, phone and order to the
lead's ``reporting_webhook_url``. With ORDER_CONFIRMATION_WEBHOOK_SECRET_KEY
unset the signer used to return "" and the webhook went out with no
``checksum`` header, which a receiver cannot tell from a forged one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Dict, List

import pytest

from app.ai.voice.agents.breeze_buddy.utils import common
from app.core.security.sha import calculate_hmac_sha256

SECRET = "webhook-signing-secret"
URL = "https://merchant.example/hook"
DATA = {"orderId": "order_1", "outcome": "CONFIRM", "name": "Asha"}


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    async def text(self) -> str:
        return ""

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Sender:
    """Stands in for ``make_request``, the guarded sender (its own tests are
    in tests/egress): records each POST and answers it with ``status``."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.posts: List[Dict[str, Any]] = []

    def __call__(self, session: Any, method: str, url: str, **kwargs: Any) -> _Response:
        self.posts.append({"method": method, "url": url, **kwargs})
        return _Response(self.status)


def _sender(monkeypatch: pytest.MonkeyPatch, status: int = 200) -> _Sender:
    sender = _Sender(status)
    monkeypatch.setattr(common, "make_request", sender)
    return sender


def test_the_signer_refuses_an_empty_key() -> None:
    # "" is not a signature. Returning it let a caller send, or compare, an
    # unsigned message as if it were signed.
    with pytest.raises(ValueError):
        calculate_hmac_sha256("payload", "")


async def test_webhook_is_not_sent_when_the_signing_key_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "ORDER_CONFIRMATION_WEBHOOK_SECRET_KEY", "")
    sender = _sender(monkeypatch)

    sent = await common.send_webhook_with_retry(None, URL, DATA)

    assert sent is False
    assert sender.posts == []


async def test_webhook_carries_the_checksum_of_its_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "ORDER_CONFIRMATION_WEBHOOK_SECRET_KEY", SECRET)
    sender = _sender(monkeypatch)

    sent = await common.send_webhook_with_retry(None, URL, DATA)

    # The receiver re-serialises the body it parsed (compact JSON) and
    # compares, so the checksum is over exactly that.
    compact = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False)
    expected = base64.b64encode(
        hmac.new(SECRET.encode(), compact.encode(), hashlib.sha256).digest()
    ).decode()
    assert sent is True
    assert len(sender.posts) == 1
    assert sender.posts[0]["method"] == "POST"
    assert sender.posts[0]["url"] == URL
    assert sender.posts[0]["headers"]["checksum"] == expected
    assert sender.posts[0]["json"] == DATA


async def test_every_retry_is_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "ORDER_CONFIRMATION_WEBHOOK_SECRET_KEY", SECRET)
    sender = _sender(monkeypatch, status=500)

    sent = await common.send_webhook_with_retry(None, URL, DATA, max_retries=3)

    assert sent is False
    assert len(sender.posts) == 3
    assert all(post["headers"].get("checksum") for post in sender.posts)
