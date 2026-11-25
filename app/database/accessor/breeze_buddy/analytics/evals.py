"""Read access for eval analytics (queries/breeze_buddy/analytics/evals.py).
Lag-tolerant reads: the replica when configured. Each call logs a failure
and re-raises it."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logger import logger
from app.database.queries import run_reader_query
from app.database.queries.breeze_buddy.analytics.evals import (
    get_eval_calls_query,
    get_eval_dashboard_query,
    get_outcome_correctness_calls_query,
    get_outcome_correctness_dashboard_query,
)
from app.utils.common import parse_json


async def get_outcome_correctness_dashboard_rows(
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        query, values = get_outcome_correctness_dashboard_query(filters)
        rows = await run_reader_query(query, values)
        return [dict(row) for row in rows or []]
    except Exception as e:
        logger.error(f"Error fetching the outcome correctness dashboard: {e}")
        raise


async def get_outcome_correctness_calls(
    filters: Dict[str, Any],
    limit: int,
    cursor_started_at: Optional[datetime] = None,
    cursor_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        query, values = get_outcome_correctness_calls_query(
            filters, limit, cursor_started_at, cursor_id
        )
        rows = await run_reader_query(query, values)
        return [dict(row) for row in rows or []]
    except Exception as e:
        logger.error(f"Error fetching outcome correctness calls: {e}")
        raise


async def get_eval_dashboard_rows(
    filters: Dict[str, Any],
    eval_name: str,
    evaluation_config_id: str,
) -> List[Dict[str, Any]]:
    try:
        query, values = get_eval_dashboard_query(
            filters, eval_name, evaluation_config_id
        )
        rows = await run_reader_query(query, values)
        return [dict(row) for row in rows or []]
    except Exception as e:
        logger.error(f"Error fetching the dashboard of eval {eval_name!r}: {e}")
        raise


async def get_eval_calls(
    filters: Dict[str, Any],
    eval_name: str,
    limit: int,
    cursor_started_at: Optional[datetime] = None,
    cursor_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        query, values = get_eval_calls_query(
            filters, eval_name, limit, cursor_started_at, cursor_id
        )
        rows = await run_reader_query(query, values)
    except Exception as e:
        logger.error(f"Error fetching calls of eval {eval_name!r}: {e}")
        raise
    results = [dict(row) for row in rows or []]
    for result in results:
        # asyncpg hands jsonb back as text
        result["answers"] = (
            parse_json({"answers": result.get("answers")}, "answers") or []
        )
    return results
