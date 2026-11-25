"""Eval analytics SQL: the preset outcome_correctness eval and an agent's
custom evals, over evaluation_result.

Every query is one agent's results over a date range: the
evaluation_result_template_time index (COMPLETED rows) and
evaluation_result_failed_time (FAILED rows, migration 084) read just that
slice. Dates are IST days, buckets IST times (as the call analytics).

The outcome check of a call is classified from stored facts only, never from
today's threshold, so a threshold change never rewrites past days:

  no_answer    the eval gave no usable answer
  no_decision  it answered NO_DECISION (none of the agent's outcomes fit)
  changed      it replaced the agent's outcome (agent_outcome, the agent's
               word, differs from outcome)
  agreed       it named the agent's own outcome
  kept         it named another outcome, and the agent's stayed (not sure
               enough, or not saved)

Confidence bands: 0 is under 0.5, then 0.1 wide (band 5 is 0.9 to 1.0). For a
yes / no answer the confidence is the larger of P(yes) and P(no).
"""

from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.database.queries.breeze_buddy.analytics.analytics import convert_ist_to_utc
from app.database.queries.breeze_buddy.evaluation_config import OUTCOME_CORRECTNESS

CONFIDENCE_BANDS = 6

#: The outcome check's answer for "none of the agent's outcomes"
#: (app/services/evals/preset/outcome_correctness/question.py NO_DECISION).
NO_DECISION = "NO_DECISION"

#: The FAILED row error prefix of an outcome check that ran out of time
#: (conversation_analysis/preset/outcome_eval.py TIMED_OUT).
TIMED_OUT = "TIMEOUT"

CHECK_RESULTS = ("agreed", "changed", "kept", "no_decision", "no_answer")

_CHECK_RESULT = f"""
    CASE
        WHEN answer IS NULL THEN 'no_answer'
        WHEN answer = '{NO_DECISION}' THEN 'no_decision'
        WHEN changed THEN 'changed'
        WHEN lower(answer) = lower(agent_word) THEN 'agreed'
        ELSE 'kept'
    END
"""


def _band(value: str) -> str:
    """``value`` (numeric: exact, so 0.7 is band 3 as it reads; float
    width_bucket puts it in band 2) as its confidence band."""
    return f"LEAST(width_bucket({value}, 0.5, 1.0, 5), 5)"


# a custom eval's answer, as a chart counts it: yes / no from P(yes), a score
# at its nearest level, a choice as given; NULL for no answer
_ANSWER = """
    CASE
        WHEN item ->> 'value' IS NULL THEN NULL
        WHEN item ->> 'kind' = 'noul' THEN
            CASE WHEN (item ->> 'value')::float >= 0.5 THEN 'yes' ELSE 'no' END
        WHEN item ->> 'kind' = 'score' THEN
            round((item ->> 'value')::numeric)::int::text
        ELSE item ->> 'value'
    END
"""

_ANSWER_BAND = f"""
    CASE
        WHEN item ->> 'kind' = 'noul' AND item ->> 'value' IS NOT NULL THEN
            {_band("GREATEST((item ->> 'value')::numeric, 1 - (item ->> 'value')::numeric)")}
        WHEN item ->> 'confidence' IS NOT NULL THEN
            {_band("(item ->> 'confidence')::numeric")}
    END
"""

#: Bands under 0.80: an unsure answer.
UNSURE_BELOW_BAND = 4


def bucket_size(date_from: date, date_to: date) -> timedelta:
    """15 minutes for one day, an hour for two, a day beyond: a chart keeps
    to a readable number of columns (96, 48, then one per day)."""
    days = (date_to - date_from).days + 1
    if days <= 1:
        return timedelta(minutes=15)
    if days <= 2:
        return timedelta(hours=1)
    return timedelta(days=1)


class _Values:
    """The query's parameter list: ``add`` returns a value's placeholder."""

    def __init__(self) -> None:
        self.values: List[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"


def _scope(filters: Dict[str, Any], params: _Values) -> List[str]:
    """One agent's results in the caller's tenant scope over IST days."""
    clauses = [f"er.template_id = {params.add(str(filters['template_id']))}::uuid"]
    if filters.get("reseller_id"):
        clauses.append(f"er.reseller_id = {params.add(filters['reseller_id'])}")
    if filters.get("reseller_ids"):
        clauses.append(
            f"er.reseller_id = ANY({params.add(filters['reseller_ids'])}::text[])"
        )
    if filters.get("merchant_id"):
        clauses.append(f"er.merchant_id = {params.add(filters['merchant_id'])}")
    if filters.get("merchant_ids"):
        clauses.append(
            f"er.merchant_id = ANY({params.add(filters['merchant_ids'])}::text[])"
        )
    start = convert_ist_to_utc(datetime.combine(filters["date_from"], time.min))
    end = convert_ist_to_utc(
        datetime.combine(filters["date_to"] + timedelta(days=1), time.min)
    )
    clauses.append(f"er.started_at >= {params.add(start)}")
    clauses.append(f"er.started_at < {params.add(end)}")
    return clauses


def _bucket(filters: Dict[str, Any], params: _Values, column: str) -> str:
    """The IST bucket ``column`` (a timestamptz) falls in, as a naive IST
    timestamp; buckets start at date_from's IST midnight."""
    # date_bin by hand: it needs Postgres 14
    size = bucket_size(filters["date_from"], filters["date_to"])
    seconds = params.add(size.total_seconds())
    step = params.add(size)
    origin = params.add(datetime.combine(filters["date_from"], time.min))
    return (
        f"({origin}::timestamp + floor(EXTRACT(EPOCH FROM ("
        f"{column} AT TIME ZONE 'Asia/Kolkata' - {origin}::timestamp"
        f")) / {seconds}::float) * {step}::interval)"
    )


def _outcome_checks(filters: Dict[str, Any], params: _Values) -> str:
    """Each outcome check in scope with the facts it is classified by."""
    where = " AND ".join(_scope(filters, params))
    name = params.add(OUTCOME_CORRECTNESS)
    return f"""
        SELECT er.id, er.source_id, er.started_at,
               answer.value AS answer, answer.confidence,
               COALESCE(lt.agent_outcome, lt.outcome) AS agent_word,
               (lt.agent_outcome IS NOT NULL
                AND lt.agent_outcome IS DISTINCT FROM lt.outcome) AS changed,
               lt.call_id,
               EXTRACT(EPOCH FROM (lt.call_end_time - lt.call_initiated_time))
                   AS duration_seconds,
               lt.meta_data ->> 'call_ended_by' AS ended_by
        FROM evaluation_result er
        JOIN lead_call_tracker lt ON lt.id = er.source_id
        LEFT JOIN LATERAL (
            SELECT item ->> 'value' AS value,
                   (item ->> 'confidence')::numeric AS confidence
            FROM jsonb_array_elements(er.metadata -> 'result') AS item
            WHERE item ->> 'key' = 'outcome'
            LIMIT 1
        ) answer ON true
        WHERE {where}
          AND er.status = 'COMPLETED'
          AND er.evaluation_type = 'CONVERSATION_EVALS'
          AND er.result = {name}
    """


def get_outcome_correctness_dashboard_query(
    filters: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Counts by IST bucket and result, with the agent's and the eval's
    outcomes and the confidence band where they disagreed, plus the checks
    that failed or ran out of time (``failed`` / ``timed_out``) — one row
    per group, aggregated by the caller."""
    params = _Values()
    checks = _outcome_checks(filters, params)
    failed_where = " AND ".join(_scope(filters, params))
    name = params.add(OUTCOME_CORRECTNESS)
    timed_out = params.add(f"{TIMED_OUT}%")
    query = f"""
        WITH checks AS ({checks}),
        classified AS (
            SELECT started_at, answer, confidence, agent_word,
                   {_CHECK_RESULT} AS check_result
            FROM checks
        )
        SELECT {_bucket(filters, params, "started_at")} AS bucket,
               check_result,
               CASE WHEN check_result IN ('no_decision', 'kept', 'changed')
                    THEN agent_word END AS agent_outcome,
               CASE WHEN check_result IN ('kept', 'changed')
                    THEN answer END AS eval_outcome,
               CASE WHEN check_result IN ('kept', 'changed')
                         AND confidence IS NOT NULL
                    THEN {_band("confidence")} END AS band,
               count(*)::int AS checks,
               (sum(confidence) FILTER (
                   WHERE check_result IN ('kept', 'changed')
               ))::float AS confidence_sum
        FROM classified
        GROUP BY 1, 2, 3, 4, 5
        UNION ALL
        SELECT {_bucket(filters, params, "er.started_at")} AS bucket,
               CASE WHEN er.error_message LIKE {timed_out}
                    THEN 'timed_out' ELSE 'failed' END AS check_result,
               NULL, NULL, NULL,
               count(*)::int AS checks,
               NULL
        FROM evaluation_result er
        WHERE {failed_where}
          AND er.status = 'FAILED'
          AND er.evaluation_type = 'CONVERSATION_EVALS'
          AND er.evaluation_config_id = (
              SELECT id FROM evaluation_config
              WHERE template_id IS NULL AND name = {name}
          )
          -- a check cancelled right as it stored its verdict counts once
          AND NOT EXISTS (
              SELECT 1 FROM evaluation_result done
              WHERE done.source_id = er.source_id
                AND done.evaluation_type = er.evaluation_type
                AND done.result = {name}
                AND done.status = 'COMPLETED'
          )
        GROUP BY 1, 2
    """
    return query, params.values


def get_outcome_correctness_calls_query(
    filters: Dict[str, Any],
    limit: int,
    cursor_started_at: Optional[datetime] = None,
    cursor_id: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """One page of outcome checks, newest first, optionally of one result
    (``check_result``) and one agent → eval pair."""
    params = _Values()
    checks = _outcome_checks(filters, params)
    clauses: List[str] = []
    if filters.get("check_result"):
        clauses.append(f"check_result = {params.add(filters['check_result'])}")
    if filters.get("agent_outcome"):
        clauses.append(f"agent_word = {params.add(filters['agent_outcome'])}")
    if filters.get("eval_outcome"):
        clauses.append(f"answer = {params.add(filters['eval_outcome'])}")
    if cursor_started_at is not None and cursor_id is not None:
        clauses.append(
            f"(started_at, id) < ({params.add(cursor_started_at)}::timestamptz, "
            f"{params.add(cursor_id)}::uuid)"
        )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        WITH checks AS ({checks}),
        classified AS (
            SELECT *, {_CHECK_RESULT} AS check_result FROM checks
        )
        SELECT id, source_id, started_at, call_id, check_result,
               agent_word AS agent_outcome, answer AS eval_outcome, confidence,
               duration_seconds, ended_by
        FROM classified
        {where}
        ORDER BY started_at DESC, id DESC
        LIMIT {params.add(limit + 1)}
    """
    return query, params.values


def get_eval_dashboard_query(
    filters: Dict[str, Any],
    eval_name: str,
    evaluation_config_id: str,
) -> Tuple[str, List[Any]]:
    """A custom eval's answers counted per question, answer and confidence
    band (``row_type`` 'answer'), its evaluated calls ('calls') and the
    calls it kept failing on ('failed')."""
    params = _Values()
    where = " AND ".join(_scope(filters, params))
    name = params.add(eval_name)
    config_id = params.add(evaluation_config_id)
    query = f"""
        SELECT 'answer' AS row_type,
               item ->> 'key' AS question_key,
               max(item ->> 'label') AS label,
               item ->> 'kind' AS kind,
               {_ANSWER} AS answer,
               {_ANSWER_BAND} AS band,
               count(*)::int AS count,
               (sum((item ->> 'value')::float) FILTER (
                   WHERE item ->> 'kind' = 'score'
               ))::float AS value_sum,
               min((item ->> 'min')::float) AS scale_min,
               max((item ->> 'max')::float) AS scale_max
        FROM evaluation_result er
        CROSS JOIN LATERAL jsonb_array_elements(er.metadata -> 'result') AS item
        WHERE {where}
          AND er.status = 'COMPLETED'
          AND er.evaluation_type = 'CONVERSATION_EVALS'
          AND er.result = {name}
        GROUP BY 2, 4, 5, 6
        UNION ALL
        SELECT 'calls', NULL, NULL, NULL, NULL, NULL, count(*)::int,
               NULL, NULL, NULL
        FROM evaluation_result er
        WHERE {where}
          AND er.status = 'COMPLETED'
          AND er.evaluation_type = 'CONVERSATION_EVALS'
          AND er.result = {name}
        UNION ALL
        SELECT 'failed', NULL, NULL, NULL, NULL, NULL, count(*)::int,
               NULL, NULL, NULL
        FROM evaluation_result er
        WHERE {where}
          AND er.status = 'FAILED'
          AND er.evaluation_type = 'CONVERSATION_EVALS'
          AND er.evaluation_config_id = {config_id}::uuid
    """
    return query, params.values


def get_eval_calls_query(
    filters: Dict[str, Any],
    eval_name: str,
    limit: int,
    cursor_started_at: Optional[datetime] = None,
    cursor_id: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """One page of a custom eval's results, newest first; optionally only
    those where a question (``question_key``) got an answer (``answer``,
    as the dashboard counts it), and only those with an unsure answer
    (``only_unsure``; to that question when one is given)."""
    params = _Values()
    clauses = _scope(filters, params)
    clauses += [
        "er.status = 'COMPLETED'",
        "er.evaluation_type = 'CONVERSATION_EVALS'",
        f"er.result = {params.add(eval_name)}",
    ]
    # the question's key is a parameter only when a clause below uses it: a
    # bound value no clause references fails the whole query
    answer_wanted = filters.get("answer") is not None
    question = (
        f"item ->> 'key' = {params.add(filters['question_key'])}"
        if filters.get("question_key") and (answer_wanted or filters.get("only_unsure"))
        else "true"
    )
    if answer_wanted:
        clauses.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements(er.metadata -> 'result') "
            f"AS item WHERE {question} AND ({_ANSWER}) = {params.add(filters['answer'])})"
        )
    if filters.get("only_unsure"):
        clauses.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements(er.metadata -> 'result') "
            f"AS item WHERE {question} "
            f"AND ({_ANSWER_BAND}) < {params.add(UNSURE_BELOW_BAND)})"
        )
    if cursor_started_at is not None and cursor_id is not None:
        clauses.append(
            f"(er.started_at, er.id) < "
            f"({params.add(cursor_started_at)}::timestamptz, "
            f"{params.add(cursor_id)}::uuid)"
        )
    query = f"""
        SELECT er.id, er.source_id, er.started_at,
               er.metadata -> 'result' AS answers
        FROM evaluation_result er
        WHERE {' AND '.join(clauses)}
        ORDER BY er.started_at DESC, er.id DESC
        LIMIT {params.add(limit + 1)}
    """
    return query, params.values
