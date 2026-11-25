"""The eval settings API: the preset outcome_correctness eval per agent (and
the platform default), and the agent's custom evals (made and managed by
anyone with access to the agent, never deleted)."""

import json
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routers.breeze_buddy.evaluations.custom import handlers as custom_evals
from app.api.routers.breeze_buddy.evaluations.preset import outcome_correctness
from app.database.queries.breeze_buddy.evaluation_config import (
    OUTCOME_CORRECTNESS,
    create_custom_eval_query,
    get_custom_eval_query,
    get_outcome_correctness_query,
    list_custom_evals_query,
    save_outcome_correctness_setting_query,
    set_custom_eval_enabled_query,
    update_custom_eval_configuration_query,
    update_outcome_correctness_default_query,
)
from app.schemas.breeze_buddy.auth import UserRole
from app.schemas.breeze_buddy.evals import (
    CreateCustomEvalRequest,
    SetCustomEvalEnabledRequest,
    UpdateCustomEvalRequest,
    UpdateOutcomeCorrectnessRequest,
)

TEMPLATE_ID = "00000000-0000-0000-0000-000000000001"
EVAL_ID = "00000000-0000-0000-0000-0000000000e1"

PRESET = {
    "engine": "structured",
    "provider": "typesafe",
    "model": "jev-1.13.0",
    "questions": [],
    "min_confidence": 0.8,
}

QUALITY = {
    "engine": "structured",
    "provider": "typesafe",
    "model": "jev-1.13.0",
    "questions": [
        {
            "type": "choice",
            "key": "tone",
            "label": "Customer tone",
            "instructions": "How did the customer sound?",
            "criteria": {"CALM": "calm", "UPSET": "upset"},
        }
    ],
}


def _settings(
    own_enabled: Optional[bool] = None, own: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "enabled": False,
        "configuration": json.dumps(PRESET),
        "own_enabled": own_enabled,
        "own_configuration": json.dumps(own) if own is not None else None,
    }


def _eval_row(enabled: bool = False, configuration: Any = None) -> Dict[str, Any]:
    return {
        "id": EVAL_ID,
        "template_id": TEMPLATE_ID,
        "evaluation_type": "CONVERSATION_EVALS",
        "name": "quality",
        "enabled": enabled,
        "configuration": json.dumps(configuration or QUALITY),
    }


@pytest.fixture
def access(monkeypatch):
    """Template access granted; returns the per-module access checks."""
    checks = SimpleNamespace(outcome=AsyncMock(), custom=AsyncMock())
    monkeypatch.setattr(outcome_correctness, "_validate_access", checks.outcome)
    monkeypatch.setattr(custom_evals, "_validate_access", checks.custom)
    return checks


#: A merchant user (not an admin) with access to the agent.
MERCHANT: Any = SimpleNamespace(role=UserRole.MERCHANT, username="merchant")
#: An admin.
ADMIN: Any = SimpleNamespace(role=UserRole.ADMIN, username="admin")


# --- outcome_correctness: the runtime and the settings SQL -------------------------


def test_the_agents_own_min_confidence_replaces_the_presets_at_runtime():
    query, values = get_outcome_correctness_query(TEMPLATE_ID)
    assert "own.configuration -> 'min_confidence' IS NOT NULL" in query
    assert "'min_confidence', own.configuration -> 'min_confidence'" in query
    # whether it runs stays the agent's row's, else the preset's
    assert "COALESCE(own.enabled, builtin.enabled)" in query
    assert values == [TEMPLATE_ID, OUTCOME_CORRECTNESS]


def test_an_agent_setting_creates_its_row_or_changes_it():
    query, values = save_outcome_correctness_setting_query(
        TEMPLATE_ID, None, {"min_confidence": 0.7}, []
    )
    assert "INSERT INTO evaluation_config" in query
    # a new row starts on the preset's flag and carries what the runtime CHECK
    # needs; an existing row keeps its flag unless one is sent
    assert "COALESCE($2::boolean, builtin.enabled)" in query
    assert "'questions', '[]'::jsonb" in query
    assert "ON CONFLICT (template_id, name) DO UPDATE" in query
    assert "COALESCE($2::boolean, evaluation_config.enabled)" in query
    # cleared keys go: the agent follows the preset again
    assert "- $4::text[]" in query
    assert values[0] == TEMPLATE_ID and values[1] is None
    assert json.loads(values[2]) == {"min_confidence": 0.7}
    assert values[3:] == [[], OUTCOME_CORRECTNESS]


def test_the_default_changes_only_the_preset_row():
    query, values = update_outcome_correctness_default_query(True, {"model": "jev-2"})
    assert "WHERE template_id IS NULL" in query
    assert "INSERT" not in query.upper()
    assert values[0] is True and json.loads(values[1]) == {"model": "jev-2"}


# --- outcome_correctness: the view -----------------------------------------------


def test_an_agent_without_its_own_row_follows_the_default():
    view = outcome_correctness.outcome_correctness_view(_settings(), TEMPLATE_ID)
    assert (view.enabled, view.min_confidence, view.model) == (False, 0.8, "jev-1.13.0")
    assert view.defaults.min_confidence == 0.8


def test_an_agents_own_settings_win():
    view = outcome_correctness.outcome_correctness_view(
        _settings(own_enabled=True, own={"min_confidence": 0.65}), TEMPLATE_ID
    )
    assert (view.enabled, view.min_confidence) == (True, 0.65)
    assert (view.defaults.enabled, view.defaults.min_confidence) == (False, 0.8)


def test_an_agent_row_without_a_threshold_follows_the_default_one():
    view = outcome_correctness.outcome_correctness_view(
        _settings(own_enabled=True, own={"engine": "structured"}), TEMPLATE_ID
    )
    assert (view.enabled, view.min_confidence) == (True, 0.8)


# --- outcome_correctness: the handlers ---------------------------------------------


async def test_an_agent_turns_it_on(monkeypatch, access):
    save = AsyncMock(return_value={"id": "x"})
    monkeypatch.setattr(outcome_correctness, "save_outcome_correctness_setting", save)
    monkeypatch.setattr(
        outcome_correctness,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings(own_enabled=True)),
    )

    view = await outcome_correctness.update_outcome_correctness_handler(
        TEMPLATE_ID, UpdateOutcomeCorrectnessRequest(enabled=True), MERCHANT
    )

    save.assert_awaited_once_with(TEMPLATE_ID, True, {}, [])
    access.outcome.assert_awaited_once()
    assert view.enabled is True


async def test_an_admin_sets_and_clears_an_agents_threshold(monkeypatch, access):
    save = AsyncMock(return_value={"id": "x"})
    monkeypatch.setattr(outcome_correctness, "save_outcome_correctness_setting", save)
    monkeypatch.setattr(
        outcome_correctness,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings()),
    )

    await outcome_correctness.update_outcome_correctness_handler(
        TEMPLATE_ID, UpdateOutcomeCorrectnessRequest(min_confidence=0.7), ADMIN
    )
    await outcome_correctness.update_outcome_correctness_handler(
        TEMPLATE_ID,
        UpdateOutcomeCorrectnessRequest.model_validate({"min_confidence": None}),
        ADMIN,
    )

    first, second = save.await_args_list
    assert first.args == (TEMPLATE_ID, None, {"min_confidence": 0.7}, [])
    assert second.args == (TEMPLATE_ID, None, {}, ["min_confidence"])


@pytest.mark.parametrize("body", [{"min_confidence": 0.7}, {"min_confidence": None}])
async def test_only_an_admin_edits_an_agents_threshold(monkeypatch, access, body):
    save = AsyncMock()
    monkeypatch.setattr(outcome_correctness, "save_outcome_correctness_setting", save)

    with pytest.raises(HTTPException) as exc:
        await outcome_correctness.update_outcome_correctness_handler(
            TEMPLATE_ID, UpdateOutcomeCorrectnessRequest.model_validate(body), MERCHANT
        )

    assert exc.value.status_code == 403
    save.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [
        {},  # nothing to change
        {"enabled": None},  # on or off, never null
        {"model": "jev-2"},  # the model is the platform default's
    ],
)
async def test_an_agent_update_is_refused(monkeypatch, access, body):
    save = AsyncMock()
    monkeypatch.setattr(outcome_correctness, "save_outcome_correctness_setting", save)

    with pytest.raises(HTTPException) as exc:
        await outcome_correctness.update_outcome_correctness_handler(
            TEMPLATE_ID, UpdateOutcomeCorrectnessRequest.model_validate(body), MERCHANT
        )

    assert exc.value.status_code == 400
    save.assert_not_awaited()


@pytest.mark.parametrize("value", [0, 1.5, -0.1])
def test_a_threshold_is_above_zero_and_at_most_one(value):
    with pytest.raises(ValidationError):
        UpdateOutcomeCorrectnessRequest(min_confidence=value)


async def test_the_default_is_admins_only(monkeypatch):
    def refuse(user):
        raise HTTPException(status_code=403, detail="admins only")

    monkeypatch.setattr(outcome_correctness, "require_admin", refuse)
    update = AsyncMock()
    monkeypatch.setattr(
        outcome_correctness, "update_outcome_correctness_default", update
    )

    with pytest.raises(HTTPException) as exc:
        await outcome_correctness.update_outcome_correctness_handler(
            "default", UpdateOutcomeCorrectnessRequest(enabled=True), MERCHANT
        )
    assert exc.value.status_code == 403
    update.assert_not_awaited()


async def test_an_admin_changes_the_default(monkeypatch):
    monkeypatch.setattr(outcome_correctness, "require_admin", lambda user: None)
    update = AsyncMock(return_value={"id": "x"})
    monkeypatch.setattr(
        outcome_correctness, "update_outcome_correctness_default", update
    )
    settings = AsyncMock(return_value=_settings())
    monkeypatch.setattr(
        outcome_correctness, "get_outcome_correctness_settings", settings
    )

    view = await outcome_correctness.update_outcome_correctness_handler(
        "default",
        UpdateOutcomeCorrectnessRequest(min_confidence=0.9, model="jev-2"),
        MERCHANT,
    )

    update.assert_awaited_once_with(None, {"min_confidence": 0.9, "model": "jev-2"})
    settings.assert_awaited_with(None)
    assert view.template_id is None


# --- custom evals: the SQL -------------------------------------------------------


def test_custom_evals_are_the_agents_rows_but_the_presets():
    for query, values in (
        list_custom_evals_query(TEMPLATE_ID),
        get_custom_eval_query(TEMPLATE_ID, "quality"),
        update_custom_eval_configuration_query(TEMPLATE_ID, "quality", QUALITY),
        set_custom_eval_enabled_query(TEMPLATE_ID, "quality", True),
    ):
        assert "evaluation_type = 'CONVERSATION_EVALS'" in query
        # not the preset's row
        assert "name <> ALL($" in query
        assert [OUTCOME_CORRECTNESS] in values
        assert "DELETE" not in query.upper()


def test_a_new_eval_never_replaces_one_of_the_same_name():
    query, values = create_custom_eval_query(TEMPLATE_ID, "quality", QUALITY, False)
    assert "ON CONFLICT (template_id, name) DO NOTHING" in query
    assert values[:3] == [TEMPLATE_ID, "quality", False]
    assert json.loads(values[3]) == QUALITY


# --- custom evals: the handlers ------------------------------------------------------


async def test_anyone_with_access_makes_an_eval(monkeypatch, access):
    create = AsyncMock(return_value=_eval_row())
    monkeypatch.setattr(custom_evals, "create_custom_eval", create)

    response = await custom_evals.create_custom_eval_handler(
        TEMPLATE_ID,
        CreateCustomEvalRequest(name="quality", configuration=QUALITY),
        MERCHANT,
    )

    access.custom.assert_awaited_once_with(TEMPLATE_ID, MERCHANT)
    create.assert_awaited_once_with(TEMPLATE_ID, "quality", QUALITY, False)
    assert (response.name, response.enabled) == ("quality", False)
    assert response.configuration == QUALITY  # merchants see their questions


@pytest.mark.parametrize("name", ["topic", OUTCOME_CORRECTNESS])
async def test_a_reserved_name_is_refused(monkeypatch, access, name):
    create = AsyncMock()
    monkeypatch.setattr(custom_evals, "create_custom_eval", create)

    with pytest.raises(HTTPException) as exc:
        await custom_evals.create_custom_eval_handler(
            TEMPLATE_ID,
            CreateCustomEvalRequest(name=name, configuration=QUALITY),
            MERCHANT,
        )

    assert exc.value.status_code == 400
    create.assert_not_awaited()


@pytest.mark.parametrize("name", ["Quality", "1st", "has-dash", "a" * 65, ""])
def test_a_name_is_lowercase_snake(name):
    with pytest.raises(ValidationError):
        CreateCustomEvalRequest(name=name, configuration=QUALITY)


@pytest.mark.parametrize(
    "configuration",
    [
        {**QUALITY, "questions": []},  # nothing to ask
        {**QUALITY, "engine": "unknown"},
        # the prompt engine's evals do not run yet
        {
            "engine": "prompt",
            "provider": "openrouter",
            "model": "m",
            "questions": [
                {
                    "type": "noul",
                    "key": "a",
                    "label": "A",
                    "instructions": "Did it?",
                }
            ],
        },
    ],
)
async def test_an_eval_its_engine_cannot_run_is_refused(
    monkeypatch, access, configuration
):
    create = AsyncMock()
    monkeypatch.setattr(custom_evals, "create_custom_eval", create)

    with pytest.raises(HTTPException) as exc:
        await custom_evals.create_custom_eval_handler(
            TEMPLATE_ID,
            CreateCustomEvalRequest(name="quality", configuration=configuration),
            MERCHANT,
        )

    assert exc.value.status_code == 400
    create.assert_not_awaited()


async def test_a_taken_name_is_a_conflict(monkeypatch, access):
    monkeypatch.setattr(
        custom_evals, "create_custom_eval", AsyncMock(return_value=None)
    )

    with pytest.raises(HTTPException) as exc:
        await custom_evals.create_custom_eval_handler(
            TEMPLATE_ID,
            CreateCustomEvalRequest(name="quality", configuration=QUALITY),
            MERCHANT,
        )

    assert exc.value.status_code == 409


async def test_an_eval_is_listed_edited_and_turned_off(monkeypatch, access):
    monkeypatch.setattr(
        custom_evals,
        "list_custom_evals",
        AsyncMock(return_value=[_eval_row(enabled=True)]),
    )
    update = AsyncMock(return_value=_eval_row(enabled=True))
    monkeypatch.setattr(custom_evals, "update_custom_eval_configuration", update)
    flip = AsyncMock(return_value=_eval_row(enabled=False))
    monkeypatch.setattr(custom_evals, "set_custom_eval_enabled", flip)

    listing = await custom_evals.list_custom_evals_handler(TEMPLATE_ID, MERCHANT)
    assert [entry.name for entry in listing.evals] == ["quality"]

    with pytest.raises(HTTPException) as exc:
        await custom_evals.update_custom_eval_handler(
            TEMPLATE_ID,
            "quality",
            UpdateCustomEvalRequest(configuration=QUALITY),
            MERCHANT,
        )
    assert exc.value.status_code == 403
    update.assert_not_awaited()

    await custom_evals.update_custom_eval_handler(
        TEMPLATE_ID, "quality", UpdateCustomEvalRequest(configuration=QUALITY), ADMIN
    )
    update.assert_awaited_once_with(TEMPLATE_ID, "quality", QUALITY)

    turned_off = await custom_evals.set_custom_eval_enabled_handler(
        TEMPLATE_ID, "quality", SetCustomEvalEnabledRequest(enabled=False), MERCHANT
    )
    flip.assert_awaited_once_with(TEMPLATE_ID, "quality", False)
    assert turned_off.enabled is False


@pytest.mark.parametrize("call", ["get", "update", "enable"])
async def test_an_unknown_eval_is_not_found(monkeypatch, access, call):
    for accessor in (
        "get_custom_eval",
        "update_custom_eval_configuration",
        "set_custom_eval_enabled",
    ):
        monkeypatch.setattr(custom_evals, accessor, AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc:
        if call == "get":
            await custom_evals.get_custom_eval_handler(TEMPLATE_ID, "nope", MERCHANT)
        elif call == "update":
            await custom_evals.update_custom_eval_handler(
                TEMPLATE_ID,
                "nope",
                UpdateCustomEvalRequest(configuration=QUALITY),
                ADMIN,
            )
        else:
            await custom_evals.set_custom_eval_enabled_handler(
                TEMPLATE_ID, "nope", SetCustomEvalEnabledRequest(enabled=True), MERCHANT
            )

    assert exc.value.status_code == 404
