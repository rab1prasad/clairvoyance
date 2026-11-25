"""Eval analytics: the agent analytics Evals tab.

  outcome-correctness-dashboard  the preset eval's checks over a date range:
                                 totals by result, the trend in IST buckets
                                 (15 min for a day, hourly for two, daily),
                                 the agent → eval pairs it disagreed on, the
                                 agent outcomes disagreement starts from, and
                                 how sure it was when it disagreed
  outcome-correctness-calls      one page of its checks (filterable by result
                                 and pair)
  eval-dashboard                 a custom eval's answers per question
  eval-calls                     one page of a custom eval's results

Every type is one agent (template_id) over IST days. An eval that is off for
the agent shows no data at all: the outcome dashboard says it is off, and a
disabled custom eval is not found. The aggregation below is pure; the SQL
returns rows already grouped (queries/breeze_buddy/analytics/evals.py).
"""

from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import HTTPException, status

from app.api.routers.breeze_buddy.evaluations.handlers import _validate_access
from app.api.routers.breeze_buddy.evaluations.preset.outcome_correctness import (
    outcome_correctness_view,
)
from app.database.accessor.breeze_buddy.analytics.evals import (
    get_eval_calls,
    get_eval_dashboard_rows,
    get_outcome_correctness_calls,
    get_outcome_correctness_dashboard_rows,
)
from app.database.accessor.breeze_buddy.evaluation_config import (
    get_custom_eval,
    get_outcome_correctness_settings,
)
from app.database.queries.breeze_buddy.analytics.evals import (
    CHECK_RESULTS,
    CONFIDENCE_BANDS,
    UNSURE_BELOW_BAND,
    bucket_size,
)
from app.schemas import UserInfo
from app.utils.common import parse_json

from .handlers import _validate_topic_filters

IST = ZoneInfo("Asia/Kolkata")

#: The longest date range one request may read.
MAX_RANGE_DAYS = 92

#: Every outcome-check result a dashboard counts: the stored checks', and
#: the checks that ran out of time or failed.
DASHBOARD_RESULTS = CHECK_RESULTS + ("timed_out", "failed")

# the results where the eval did not name the agent's outcome
_DISAGREED = ("no_decision", "kept", "changed")
# the results where it named another of the agent's outcomes
_OTHER_OUTCOME = ("kept", "changed")


def _validate_eval_filters(filters: Dict[str, Any], *, custom: bool) -> str:
    """Dates, one agent and (for a custom eval) its name. Returns the
    template id."""
    _validate_topic_filters(filters)
    missing = [key for key in ("template_id",) if not filters.get(key)]
    if custom and not filters.get("eval_name"):
        missing.append("eval_name")
    if missing:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Missing {', '.join(missing)}"
        )
    if (filters["date_to"] - filters["date_from"]).days + 1 > MAX_RANGE_DAYS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"The date range is at most {MAX_RANGE_DAYS} days",
        )
    return str(filters["template_id"])


def _page(options: Dict[str, Any]) -> Tuple[int, Optional[datetime], Optional[str]]:
    """The page size and the cursor (``started_at|id``) to read after."""
    limit = max(1, min(options.get("limit", 50), 100))
    cursor = options.get("cursor")
    if not cursor:
        return limit, None, None
    try:
        started_at, row_id = cursor.split("|", 1)
        return limit, datetime.fromisoformat(started_at), str(UUID(row_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor"
        ) from exc


def _next_cursor(rows: List[Dict[str, Any]], limit: int) -> Optional[str]:
    if len(rows) <= limit:
        return None
    last = rows[limit - 1]
    return f"{last['started_at'].isoformat()}|{last['id']}"


def band_range(band: int) -> Tuple[float, float]:
    """A confidence band's bounds: band 0 is under 0.5, then 0.1 wide."""
    if band == 0:
        return 0.0, 0.5
    return round(0.4 + 0.1 * band, 1), round(0.5 + 0.1 * band, 1)


def _ist(bucket: datetime) -> str:
    """A bucket (a naive IST timestamp from the query) as ISO with +05:30."""
    return bucket.replace(tzinfo=IST).isoformat()


# --- outcome_correctness -------------------------------------------------------


def outcome_correctness_dashboard(
    rows: Sequence[Mapping[str, Any]], date_from: date, date_to: date
) -> Dict[str, Any]:
    """PURE: the dashboard from the grouped rows of
    get_outcome_correctness_dashboard_query."""
    size = bucket_size(date_from, date_to)
    totals = {result: 0 for result in DASHBOARD_RESULTS}
    trend: Dict[datetime, Dict[str, int]] = {}
    pairs: Dict[Tuple[str, str, str], List[float]] = {}
    by_agent: Dict[str, Dict[str, int]] = {}
    bands = [{"kept": 0, "changed": 0} for _ in range(CONFIDENCE_BANDS)]

    for row in rows:
        result, count = row["check_result"], int(row["checks"])
        totals[result] += count
        bucket = trend.setdefault(
            row["bucket"], {name: 0 for name in DASHBOARD_RESULTS}
        )
        bucket[result] += count
        if result in _DISAGREED and row.get("agent_outcome"):
            agent = by_agent.setdefault(
                row["agent_outcome"], {name: 0 for name in _DISAGREED}
            )
            agent[result] += count
        if result in _OTHER_OUTCOME:
            if row.get("eval_outcome"):
                pair = pairs.setdefault(
                    (row.get("agent_outcome") or "", row["eval_outcome"], result),
                    [0, 0.0],
                )
                pair[0] += count
                pair[1] += float(row.get("confidence_sum") or 0.0)
            if row.get("band") is not None:
                bands[int(row["band"])][result] += count

    # every bucket of the range, empty ones included, in order
    start = datetime.combine(date_from, time.min)
    end = datetime.combine(date_to + timedelta(days=1), time.min)
    buckets = []
    at = start
    while at < end:
        counts = trend.get(at) or {name: 0 for name in DASHBOARD_RESULTS}
        buckets.append({"bucket": _ist(at), **counts})
        at += size

    return {
        "bucket_minutes": int(size.total_seconds() // 60),
        "totals": {"checked": sum(totals.values()), **totals},
        "trend": buckets,
        "pairs": sorted(
            (
                {
                    "agent_outcome": agent,
                    "eval_outcome": answer,
                    "check_result": result,
                    "calls": int(count),
                    "average_confidence": (
                        round(confidence / count, 3) if count else None
                    ),
                }
                for (agent, answer, result), (count, confidence) in pairs.items()
            ),
            key=lambda pair: (-pair["calls"], pair["agent_outcome"]),
        ),
        "by_agent_outcome": sorted(
            (
                {"agent_outcome": agent, "total": sum(counts.values()), **counts}
                for agent, counts in by_agent.items()
            ),
            key=lambda row: (-row["total"], row["agent_outcome"]),
        ),
        "confidence": [
            {
                "band": band,
                "from": band_range(band)[0],
                "to": band_range(band)[1],
                **counts,
            }
            for band, counts in enumerate(bands)
        ],
    }


async def _outcome_settings(template_id: str) -> Dict[str, Any]:
    row = await get_outcome_correctness_settings(template_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The outcome_correctness preset is not set up",
        )
    return outcome_correctness_view(row, template_id).model_dump(mode="json")


async def get_outcome_correctness_dashboard_analytics(
    filters: Dict[str, Any], options: Dict[str, Any], current_user: UserInfo
) -> Dict[str, Any]:
    template_id = _validate_eval_filters(filters, custom=False)
    await _validate_access(template_id, current_user)
    settings = await _outcome_settings(template_id)
    response: Dict[str, Any] = {
        "type": "outcome-correctness-dashboard",
        "filters_applied": filters,
        "settings": settings,
    }
    if not settings["enabled"]:
        return response  # off for this agent: no data shown
    rows = await get_outcome_correctness_dashboard_rows(filters)
    return {
        **response,
        **outcome_correctness_dashboard(rows, filters["date_from"], filters["date_to"]),
    }


async def get_outcome_correctness_calls_analytics(
    filters: Dict[str, Any], options: Dict[str, Any], current_user: UserInfo
) -> Dict[str, Any]:
    template_id = _validate_eval_filters(filters, custom=False)
    await _validate_access(template_id, current_user)
    settings = await _outcome_settings(template_id)
    limit, cursor_started_at, cursor_id = _page(options)
    rows = (
        await get_outcome_correctness_calls(
            filters, limit, cursor_started_at, cursor_id
        )
        if settings["enabled"]
        else []
    )
    return {
        "type": "outcome-correctness-calls",
        "filters_applied": filters,
        "results": [
            {
                "evaluation_id": str(row["id"]),
                "source_id": row["source_id"],
                "call_id": row.get("call_id"),
                "started_at": row["started_at"],
                "check_result": row["check_result"],
                "agent_outcome": row.get("agent_outcome"),
                "eval_outcome": row.get("eval_outcome"),
                "confidence": (
                    float(row["confidence"])
                    if row.get("confidence") is not None
                    else None
                ),
                "duration_seconds": (
                    float(row["duration_seconds"])
                    if row.get("duration_seconds") is not None
                    else None
                ),
                "ended_by": row.get("ended_by"),
            }
            for row in rows[:limit]
        ],
        "pagination": {
            "limit": limit,
            "has_more": len(rows) > limit,
            "next_cursor": _next_cursor(rows, limit),
        },
    }


# --- custom evals ----------------------------------------------------------------


def answer_of(result: Mapping[str, Any]) -> Optional[str]:
    """PURE: a stored answer as the dashboard counts it (the SQL's _ANSWER):
    yes / no from P(yes), a score at its nearest level (halves round away
    from zero, as Postgres does), a choice as given; None for no answer."""
    value = result.get("value")
    if value is None:
        return None
    kind = result.get("kind")
    if kind == "noul":
        return "yes" if float(value) >= 0.5 else "no"
    if kind == "score":
        return str(int(Decimal(str(value)).quantize(Decimal(1), ROUND_HALF_UP)))
    return str(value)


def band_of(result: Mapping[str, Any]) -> Optional[int]:
    """PURE: a stored answer's confidence band (the SQL's _ANSWER_BAND,
    exact as numeric is: 0.7 is band 3); None when the engine gave no
    confidence."""
    if result.get("kind") == "noul":
        value = result.get("value")
        if value is None:
            return None
        p_yes = Decimal(str(value))
        sure = max(p_yes, 1 - p_yes)
    elif result.get("confidence") is not None:
        sure = Decimal(str(result["confidence"]))
    else:
        return None
    if sure < Decimal("0.5"):
        return 0
    # width_bucket(sure, 0.5, 1.0, 5), capped at 5
    return min(CONFIDENCE_BANDS - 1, 1 + int((sure - Decimal("0.5")) * 10))


def question_answers(question: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """PURE: a configured question's possible answers, as the dashboard
    counts them, with each one's description from the config: a choice's
    options, a score's levels (0 and 1 for a 2-level score, as stored; else
    1 to N), yes and no."""
    criteria = question.get("criteria")
    kind = question.get("type")
    if kind == "choice" and isinstance(criteria, dict):
        return [(str(option), str(text)) for option, text in criteria.items()]
    if kind == "score" and isinstance(criteria, list):
        first = 0 if len(criteria) == 2 else 1
        return [(str(first + index), str(text)) for index, text in enumerate(criteria)]
    if kind == "noul":
        texts = criteria if isinstance(criteria, dict) else {}
        return [("yes", texts.get("true")), ("no", texts.get("false"))]
    return []  # the prompt engine's questions name no answers


def eval_dashboard(
    rows: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any]
) -> Dict[str, Any]:
    """PURE: a custom eval's dashboard from the grouped rows of
    get_eval_dashboard_query and its current configuration. Questions come
    in config order, then any only older results hold (``in_config``
    false); each question's answers list its configured ones first (zero
    counts included), then any no longer in the config."""
    totals = {"calls": 0, "failed": 0}
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["row_type"] != "answer":
            totals[row["row_type"]] = int(row["count"] or 0)
            continue
        question = seen.setdefault(
            row["question_key"],
            {
                "label": row.get("label"),
                "kind": row.get("kind"),
                "answers": {},
                "no_answer": 0,
                "bands": [0] * CONFIDENCE_BANDS,
                "has_bands": False,
                "value_sum": 0.0,
                "scored": 0,
                "scale_min": None,
                "scale_max": None,
            },
        )
        count = int(row["count"])
        if row.get("answer") is None:
            question["no_answer"] += count
        else:
            answers = question["answers"]
            answers[row["answer"]] = answers.get(row["answer"], 0) + count
            if row.get("value_sum") is not None:
                question["value_sum"] += float(row["value_sum"])
                question["scored"] += count
        if row.get("band") is not None:
            question["bands"][int(row["band"])] += count
            question["has_bands"] = True
        for bound in ("scale_min", "scale_max"):
            if row.get(bound) is not None:
                question[bound] = float(row[bound])

    configured = [
        question
        for question in configuration.get("questions") or []
        if isinstance(question, Mapping) and question.get("key")
    ]
    keys = [str(question["key"]) for question in configured]
    by_key = {str(question["key"]): question for question in configured}
    questions = []
    for key in keys + sorted(key for key in seen if key not in by_key):
        config = by_key.get(key)
        stats = seen.get(key) or {}
        counts: Dict[str, int] = dict(stats.get("answers") or {})
        possible = question_answers(config) if config else []
        answers = [
            {
                "answer": answer,
                "count": counts.pop(answer, 0),
                "description": description,
                "in_config": True,
            }
            for answer, description in possible
        ]
        answers += [
            {"answer": answer, "count": count, "description": None, "in_config": False}
            for answer, count in sorted(counts.items(), key=lambda item: -item[1])
        ]
        answered = sum(answer["count"] for answer in answers)
        scored = stats.get("scored") or 0
        questions.append(
            {
                "key": key,
                "label": (config or {}).get("label") or stats.get("label") or key,
                "kind": (config or {}).get("type") or stats.get("kind"),
                "in_config": config is not None,
                "answers": answers,
                "answered": answered,
                "no_answer": stats.get("no_answer") or 0,
                "average": (round(stats["value_sum"] / scored, 3) if scored else None),
                "scale": (
                    {"min": stats["scale_min"], "max": stats["scale_max"]}
                    if stats.get("scale_min") is not None
                    else None
                ),
                # None when the engine gives no confidence (the prompt engine)
                "confidence": (stats["bands"] if stats.get("has_bands") else None),
                "unsure": (
                    sum(stats["bands"][:UNSURE_BELOW_BAND])
                    if stats.get("has_bands")
                    else None
                ),
            }
        )
    return {"totals": totals, "questions": questions}


async def _enabled_eval(template_id: str, name: str) -> Dict[str, Any]:
    """The agent's custom eval, if it is on: a disabled one shows nothing."""
    row = await get_custom_eval(template_id, name)
    if not row or not row.get("enabled"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enabled eval named {name!r} for this agent",
        )
    return row


async def get_eval_dashboard_analytics(
    filters: Dict[str, Any], options: Dict[str, Any], current_user: UserInfo
) -> Dict[str, Any]:
    template_id = _validate_eval_filters(filters, custom=True)
    await _validate_access(template_id, current_user)
    row = await _enabled_eval(template_id, filters["eval_name"])
    configuration = parse_json(row, "configuration") or {}
    rows = await get_eval_dashboard_rows(filters, row["name"], str(row["id"]))
    return {
        "type": "eval-dashboard",
        "filters_applied": filters,
        "eval": {"name": row["name"], "configuration": configuration},
        **eval_dashboard(rows, configuration),
    }


async def get_eval_calls_analytics(
    filters: Dict[str, Any], options: Dict[str, Any], current_user: UserInfo
) -> Dict[str, Any]:
    template_id = _validate_eval_filters(filters, custom=True)
    await _validate_access(template_id, current_user)
    row = await _enabled_eval(template_id, filters["eval_name"])
    limit, cursor_started_at, cursor_id = _page(options)
    rows = await get_eval_calls(
        filters, row["name"], limit, cursor_started_at, cursor_id
    )
    results = []
    for result_row in rows[:limit]:
        answers = []
        for answer in result_row.get("answers") or []:
            band = band_of(answer)
            answers.append(
                {
                    "key": answer.get("key"),
                    "label": answer.get("label"),
                    "kind": answer.get("kind"),
                    "value": answer.get("value"),
                    "confidence": answer.get("confidence"),
                    "answer": answer_of(answer),
                    "band": band,
                    "unsure": band is not None and band < UNSURE_BELOW_BAND,
                }
            )
        results.append(
            {
                "evaluation_id": str(result_row["id"]),
                "source_id": result_row["source_id"],
                "started_at": result_row["started_at"],
                "answers": answers,
            }
        )
    return {
        "type": "eval-calls",
        "filters_applied": filters,
        "results": results,
        "pagination": {
            "limit": limit,
            "has_more": len(rows) > limit,
            "next_cursor": _next_cursor(rows, limit),
        },
    }
