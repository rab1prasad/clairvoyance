"""Database access for per-template evaluation configuration."""

from typing import Any, Dict, List, Optional

from app.database.queries import run_parameterized_query
from app.database.queries.breeze_buddy.evaluation_config import (
    add_discovered_topics_query,
    create_custom_eval_query,
    get_custom_eval_query,
    get_enabled_evaluations_query,
    get_evaluation_config_query,
    get_outcome_correctness_query,
    get_outcome_correctness_settings_query,
    has_enabled_evaluations_query,
    initialize_evaluation_config_query,
    list_custom_evals_query,
    remove_topics_query,
    save_evaluation_configuration_query,
    save_outcome_correctness_setting_query,
    set_custom_eval_enabled_query,
    set_evaluation_enabled_query,
    update_custom_eval_configuration_query,
    update_evaluation_configuration_query,
    update_outcome_correctness_default_query,
)


async def initialize_evaluation_config(template_id: str) -> None:
    query, values = initialize_evaluation_config_query(template_id)
    await run_parameterized_query(query, values)


async def get_evaluation_config(
    template_id: str,
    evaluation_type: str,
) -> Optional[Dict[str, Any]]:
    query, values = get_evaluation_config_query(template_id, evaluation_type)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def get_enabled_evaluations(template_id: str) -> List[Dict[str, Any]]:
    query, values = get_enabled_evaluations_query(template_id)
    rows = await run_parameterized_query(query, values)
    return [dict(row) for row in rows or []]


async def get_outcome_correctness(template_id: str) -> Optional[Dict[str, Any]]:
    """The preset outcome_correctness row, or None when the eval is off for
    this agent (off by default; see the query)."""
    query, values = get_outcome_correctness_query(template_id)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def get_outcome_correctness_settings(
    template_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The preset row's flag and configuration with the agent's own (None
    without one); ``template_id`` None reads the preset alone. None when the
    preset row is missing."""
    query, values = get_outcome_correctness_settings_query(template_id)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def save_outcome_correctness_setting(
    template_id: str,
    enabled: Optional[bool],
    settings: Dict[str, Any],
    cleared: List[str],
) -> Optional[Dict[str, Any]]:
    query, values = save_outcome_correctness_setting_query(
        template_id, enabled, settings, cleared
    )
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def update_outcome_correctness_default(
    enabled: Optional[bool],
    settings: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query, values = update_outcome_correctness_default_query(enabled, settings)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def list_custom_evals(template_id: str) -> List[Dict[str, Any]]:
    query, values = list_custom_evals_query(template_id)
    rows = await run_parameterized_query(query, values)
    return [dict(row) for row in rows or []]


async def get_custom_eval(template_id: str, name: str) -> Optional[Dict[str, Any]]:
    query, values = get_custom_eval_query(template_id, name)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def create_custom_eval(
    template_id: str,
    name: str,
    configuration: Dict[str, Any],
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    """The new eval's row, or None when the agent already has that name."""
    query, values = create_custom_eval_query(template_id, name, configuration, enabled)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def update_custom_eval_configuration(
    template_id: str,
    name: str,
    configuration: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query, values = update_custom_eval_configuration_query(
        template_id, name, configuration
    )
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def set_custom_eval_enabled(
    template_id: str,
    name: str,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    query, values = set_custom_eval_enabled_query(template_id, name, enabled)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def has_enabled_evaluations(template_id: str) -> bool:
    query, values = has_enabled_evaluations_query(template_id)
    rows = await run_parameterized_query(query, values)
    return bool(rows and rows[0]["enabled"])


async def set_evaluation_enabled(
    template_id: str,
    evaluation_type: str,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    query, values = set_evaluation_enabled_query(template_id, evaluation_type, enabled)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def update_evaluation_configuration(
    template_id: str,
    evaluation_type: str,
    patch: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query, values = update_evaluation_configuration_query(
        template_id, evaluation_type, patch
    )
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def save_evaluation_configuration(
    template_id: str,
    evaluation_type: str,
    configuration: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query, values = save_evaluation_configuration_query(
        template_id, evaluation_type, configuration
    )
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def add_discovered_topics(
    template_id: str, labels: List[str]
) -> Optional[Dict[str, Any]]:
    query, values = add_discovered_topics_query(template_id, labels)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None


async def remove_topics(
    template_id: str, labels: List[str]
) -> Optional[Dict[str, Any]]:
    query, values = remove_topics_query(template_id, labels)
    rows = await run_parameterized_query(query, values)
    return dict(rows[0]) if rows else None
