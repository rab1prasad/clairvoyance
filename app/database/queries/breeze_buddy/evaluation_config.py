"""SQL for per-template evaluation configuration.

An agent can have many CONVERSATION_EVALS rows, each named, and one TOPIC row
(always named 'topic'). The per-type endpoints address the row named after
their type (``lower(evaluation_type)``). The agent's custom evals are its
CONVERSATION_EVALS rows other than the preset outcome_correctness one, each
addressed by its name.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

_CONFIG_COLUMNS = (
    "id, template_id, evaluation_type::text AS evaluation_type, name, "
    "enabled, topics, configuration"
)

#: The preset eval: a global row (the default for every agent) that an
#: agent's own row of the same name overrides, enabled or disabled.
OUTCOME_CORRECTNESS = "outcome_correctness"

#: The CONVERSATION_EVALS rows that are not the agent's custom evals: the
#: preset's.
NOT_CUSTOM_EVALS = [OUTCOME_CORRECTNESS]


def get_evaluation_config_query(
    template_id: str,
    evaluation_type: str,
) -> Tuple[str, List[Any]]:
    """The agent's own row for this type (no defaults fallback); the literal
    template id 'default' addresses the platform default row (NULL)."""
    query = f"""
        SELECT {_CONFIG_COLUMNS}
        FROM evaluation_config
        WHERE template_id IS NOT DISTINCT FROM NULLIF($1, 'default')::uuid
          AND evaluation_type = $2::evaluation_type
          AND name = lower($2::text)
    """
    return query, [template_id, evaluation_type]


def initialize_evaluation_config_query(template_id: str) -> Tuple[str, List[Any]]:
    query = """
        INSERT INTO evaluation_config (
            template_id, evaluation_type, name, enabled, configuration
        )
        SELECT
            template.id, defaults.evaluation_type, defaults.name, true,
            defaults.configuration
        FROM template
        CROSS JOIN evaluation_config defaults
        WHERE template.id = $1::uuid
          AND template.configurations
                -> 'enable_topic_evaluation' = 'true'::jsonb
          AND defaults.template_id IS NULL
          AND defaults.evaluation_type = 'TOPIC'
        ON CONFLICT (template_id, name) DO NOTHING
    """
    return query, [template_id]


def get_enabled_evaluations_query(template_id: str) -> Tuple[str, List[Any]]:
    query = """
        SELECT id, evaluation_type::text AS evaluation_type, name, topics,
               configuration, configuration ->> 'model' AS model
        FROM evaluation_config
        WHERE template_id = $1::uuid
          AND enabled
    """
    return query, [template_id]


def get_outcome_correctness_query(template_id: str) -> Tuple[str, List[Any]]:
    """The preset outcome_correctness row (its engine, model and threshold)
    when the eval is on for this agent: the agent's own row of that name
    decides, enabled or disabled; without one, the preset row's own flag
    (the default for every agent). The agent's own ``min_confidence``, when
    it set one, replaces the preset's; everything else stays the preset's."""
    query = """
        SELECT builtin.id, builtin.template_id,
               builtin.evaluation_type::text AS evaluation_type, builtin.name,
               builtin.enabled, builtin.topics,
               CASE
                   WHEN own.configuration -> 'min_confidence' IS NOT NULL
                   THEN builtin.configuration || jsonb_build_object(
                       'min_confidence', own.configuration -> 'min_confidence'
                   )
                   ELSE builtin.configuration
               END AS configuration
        FROM evaluation_config builtin
        LEFT JOIN evaluation_config own
          ON own.template_id = $1::uuid
         AND own.name = builtin.name
        WHERE builtin.template_id IS NULL
          AND builtin.name = $2
          AND COALESCE(own.enabled, builtin.enabled)
    """
    return query, [template_id, OUTCOME_CORRECTNESS]


def get_outcome_correctness_settings_query(
    template_id: Optional[str],
) -> Tuple[str, List[Any]]:
    """The preset outcome_correctness row and, for an agent, its own row of
    that name (NULLs without one), for the settings API. ``template_id``
    None reads the preset row alone."""
    query = """
        SELECT builtin.enabled, builtin.configuration,
               own.enabled AS own_enabled, own.configuration AS own_configuration
        FROM evaluation_config builtin
        LEFT JOIN evaluation_config own
          ON own.template_id = $1::uuid
         AND own.name = builtin.name
        WHERE builtin.template_id IS NULL
          AND builtin.name = $2
    """
    return query, [template_id, OUTCOME_CORRECTNESS]


def save_outcome_correctness_setting_query(
    template_id: str,
    enabled: Optional[bool],
    settings: Dict[str, Any],
    cleared: List[str],
) -> Tuple[str, List[Any]]:
    """Set the agent's own outcome_correctness settings, creating its row
    when it has none. ``enabled`` None keeps the flag (a new row starts on
    the preset's); ``settings`` (min_confidence) are written, ``cleared``
    keys removed so the agent follows the preset again. A new row carries
    the preset's engine, provider and model only because every
    CONVERSATION_EVALS row must (evaluation_config_runtime_check): the
    runtime reads the agent's flag and min_confidence alone."""
    query = f"""
        INSERT INTO evaluation_config (
            template_id, evaluation_type, name, enabled, configuration
        )
        SELECT
            $1::uuid, builtin.evaluation_type, builtin.name,
            COALESCE($2::boolean, builtin.enabled),
            (
                jsonb_build_object(
                    'engine', builtin.configuration -> 'engine',
                    'provider', builtin.configuration -> 'provider',
                    'model', builtin.configuration -> 'model',
                    'questions', '[]'::jsonb
                ) || $3::jsonb
            ) - $4::text[]
        FROM evaluation_config builtin
        WHERE builtin.template_id IS NULL
          AND builtin.name = $5
        ON CONFLICT (template_id, name) DO UPDATE
            SET enabled = COALESCE($2::boolean, evaluation_config.enabled),
                configuration = (evaluation_config.configuration || $3::jsonb)
                    - $4::text[]
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [
        template_id,
        enabled,
        json.dumps(settings),
        cleared,
        OUTCOME_CORRECTNESS,
    ]


def update_outcome_correctness_default_query(
    enabled: Optional[bool],
    settings: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Change the preset row, the default of every agent without its own
    setting. ``enabled`` None keeps the flag."""
    query = f"""
        UPDATE evaluation_config
        SET enabled = COALESCE($1::boolean, enabled),
            configuration = configuration || $2::jsonb
        WHERE template_id IS NULL
          AND name = $3
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [enabled, json.dumps(settings), OUTCOME_CORRECTNESS]


# --- the agent's custom evals ----------------------------------------------
# Its CONVERSATION_EVALS rows but the preset's (NOT_CUSTOM_EVALS),
# addressed by name. None is
# ever deleted: an eval is turned off, and a disabled eval neither runs nor
# shows in analytics.


def list_custom_evals_query(template_id: str) -> Tuple[str, List[Any]]:
    query = f"""
        SELECT {_CONFIG_COLUMNS}
        FROM evaluation_config
        WHERE template_id = $1::uuid
          AND evaluation_type = 'CONVERSATION_EVALS'
          AND name <> ALL($2::text[])
        ORDER BY name
    """
    return query, [template_id, NOT_CUSTOM_EVALS]


def get_custom_eval_query(template_id: str, name: str) -> Tuple[str, List[Any]]:
    query = f"""
        SELECT {_CONFIG_COLUMNS}
        FROM evaluation_config
        WHERE template_id = $1::uuid
          AND evaluation_type = 'CONVERSATION_EVALS'
          AND name = $2
          AND name <> ALL($3::text[])
    """
    return query, [template_id, name, NOT_CUSTOM_EVALS]


def create_custom_eval_query(
    template_id: str,
    name: str,
    configuration: Dict[str, Any],
    enabled: bool,
) -> Tuple[str, List[Any]]:
    """A new eval; nothing when the agent already has a row of that name."""
    query = f"""
        INSERT INTO evaluation_config (
            template_id, evaluation_type, name, enabled, configuration
        )
        VALUES ($1::uuid, 'CONVERSATION_EVALS', $2, $3::boolean, $4::jsonb)
        ON CONFLICT (template_id, name) DO NOTHING
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, name, enabled, json.dumps(configuration)]


def update_custom_eval_configuration_query(
    template_id: str,
    name: str,
    configuration: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Replace the eval's configuration wholesale; its flag is kept."""
    query = f"""
        UPDATE evaluation_config
        SET configuration = $3::jsonb
        WHERE template_id = $1::uuid
          AND evaluation_type = 'CONVERSATION_EVALS'
          AND name = $2
          AND name <> ALL($4::text[])
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, name, json.dumps(configuration), NOT_CUSTOM_EVALS]


def set_custom_eval_enabled_query(
    template_id: str,
    name: str,
    enabled: bool,
) -> Tuple[str, List[Any]]:
    query = f"""
        UPDATE evaluation_config
        SET enabled = $3::boolean
        WHERE template_id = $1::uuid
          AND evaluation_type = 'CONVERSATION_EVALS'
          AND name = $2
          AND name <> ALL($4::text[])
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, name, enabled, NOT_CUSTOM_EVALS]


def has_enabled_evaluations_query(template_id: str) -> Tuple[str, List[Any]]:
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM evaluation_config
            WHERE template_id = $1::uuid
              AND enabled
        ) AS enabled
    """
    return query, [template_id]


def set_evaluation_enabled_query(
    template_id: str,
    evaluation_type: str,
    enabled: bool,
) -> Tuple[str, List[Any]]:
    """Flip the agent's existing row — never creates one.

    No row (or a disabled row) means the agent does not run this
    evaluation."""
    query = f"""
        UPDATE evaluation_config
        SET enabled = $3::boolean
        WHERE template_id = $1::uuid
          AND evaluation_type = $2::evaluation_type
          AND name = lower($2::text)
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, evaluation_type, enabled]


def update_evaluation_configuration_query(
    template_id: str,
    evaluation_type: str,
    patch: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Shallow JSONB merge on the agent's existing row — never creates one.

    Each top-level key in the patch overwrites the stored one wholesale.
    TOPIC sends partial patches; CONVERSATION_EVALS's validator requires
    every key, so for it the merge amounts to a full replacement."""
    query = f"""
        UPDATE evaluation_config
        SET configuration = configuration || $3::jsonb
        WHERE template_id IS NOT DISTINCT FROM NULLIF($1, 'default')::uuid
          AND evaluation_type = $2::evaluation_type
          AND name = lower($2::text)
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, evaluation_type, json.dumps(patch)]


def save_evaluation_configuration_query(
    template_id: str,
    evaluation_type: str,
    configuration: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Create-or-replace — the row's creation point (POST).

    A missing row is born DISABLED: configuring an evaluation is not
    consenting to run it, enable is a separate flip (and it never
    creates). An existing row gets its configuration replaced wholesale
    and keeps its enabled flag."""
    query = f"""
        INSERT INTO evaluation_config (
            template_id, evaluation_type, name, enabled, configuration
        )
        VALUES ($1::uuid, $2::evaluation_type, lower($2::text), false, $3::jsonb)
        ON CONFLICT (template_id, name)
            DO UPDATE SET configuration = EXCLUDED.configuration
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, evaluation_type, json.dumps(configuration)]


def add_discovered_topics_query(
    template_id: str,
    labels: List[str],
) -> Tuple[str, List[Any]]:
    query = f"""
        UPDATE evaluation_config config
        SET topics = config.topics || ARRAY(
            SELECT label
            FROM unnest($2::text[]) AS discovered(label)
            WHERE NOT EXISTS (
                SELECT 1
                FROM unnest(config.topics) AS existing(label)
                WHERE lower(btrim(existing.label)) = lower(btrim(discovered.label))
            )
        )
        WHERE config.template_id = $1::uuid
          AND config.evaluation_type = 'TOPIC'
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, labels]


def remove_topics_query(
    template_id: str,
    labels: List[str],
) -> Tuple[str, List[Any]]:
    query = f"""
        UPDATE evaluation_config config
        SET topics = ARRAY(
            SELECT existing.label
            FROM unnest(config.topics) WITH ORDINALITY AS existing(label, position)
            WHERE lower(btrim(existing.label)) <> ALL(
                SELECT lower(btrim(removed.label))
                FROM unnest($2::text[]) AS removed(label)
            )
            ORDER BY existing.position
        )
        WHERE config.template_id = $1::uuid
          AND config.evaluation_type = 'TOPIC'
        RETURNING {_CONFIG_COLUMNS}
    """
    return query, [template_id, labels]
