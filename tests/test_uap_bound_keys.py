"""The server-owned UAP keys are named once, and the binding writes no other.

An anonymous widget session is opened without ``BOUND_TEMPLATE_VARS``
(widget/handlers.py). If the binding started writing a key that list does
not name, a visitor could plant it again; this pins the two together.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.services.preview.uap import accessor


@pytest.mark.asyncio
async def test_the_binding_writes_only_the_keys_named_as_server_owned(monkeypatch):
    patches: List[Dict[str, Any]] = []

    def query(session_id: str, patch: Dict[str, Any]):
        patches.append(patch)
        return "query", []

    async def run(_query, _values):
        return [{"id": "s1"}]

    monkeypatch.setattr(accessor, "merge_session_template_vars_query", query)
    monkeypatch.setattr(accessor, "run_parameterized_query", run)

    assert await accessor.bind_session_customer(
        "s1", "customer-1", "9999999999", rider_token="ny-token"
    )
    assert set(patches[0]) == set(accessor.BOUND_TEMPLATE_VARS)
