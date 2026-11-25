"""Eval analytics: the preset outcome_correctness eval's checks and an
agent's custom evals, for the agent analytics Evals tab. The SQL returns
grouped rows; these tests pin the SQL's shape and the pure aggregation, and
that an eval that is off shows nothing."""

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.ai.voice.agents.breeze_buddy.services.conversation_analysis.preset import (
    outcome_eval,
)
from app.api.routers.breeze_buddy.analytics import (
    _ANALYTICS_HANDLERS,
    evals as analytics,
)
from app.database.queries.breeze_buddy.analytics import evals as queries
from app.schemas.breeze_buddy.analytics import AnalyticsType
from app.services.evals.preset.outcome_correctness import NO_DECISION

TEMPLATE_ID = "00000000-0000-0000-0000-000000000001"
DAY = date(2026, 10, 8)
USER: Any = SimpleNamespace(role="merchant")


def _filters(**extra: Any) -> Dict[str, Any]:
    return {"template_id": TEMPLATE_ID, "date_from": DAY, "date_to": DAY, **extra}


# --- the SQL ---------------------------------------------------------------------


def test_the_sql_spells_the_constants_the_writers_use():
    # the queries cannot import the writers (the data layer), so they spell
    # the same words
    assert queries.NO_DECISION == NO_DECISION
    assert queries.TIMED_OUT == outcome_eval.TIMED_OUT


@pytest.mark.parametrize(
    "date_to, size",
    [(DAY, 15 * 60), (date(2026, 10, 9), 3600), (date(2026, 10, 14), 86400)],
)
def test_buckets_grow_with_the_range(date_to, size):
    assert queries.bucket_size(DAY, date_to).total_seconds() == size


def test_a_dashboard_reads_ist_days_for_one_agent_in_scope():
    query, values = queries.get_outcome_correctness_dashboard_query(
        _filters(reseller_ids=["r"], merchant_ids=["m"])
    )
    assert "er.template_id = $1::uuid" in query
    assert "er.reseller_id = ANY($2::text[])" in query
    assert "er.merchant_id = ANY($3::text[])" in query
    # 8 Oct IST is 7 Oct 18:30 to 8 Oct 18:30 UTC
    assert values[3] == datetime(2026, 10, 7, 18, 30, tzinfo=timezone.utc)
    assert values[4] == datetime(2026, 10, 8, 18, 30, tzinfo=timezone.utc)
    # classified from stored facts: the agent's word, whether it changed
    assert "COALESCE(lt.agent_outcome, lt.outcome) AS agent_word" in query
    # every outcome write sets agent_outcome but the check's: changed is
    # where the two differ; a lead from before the column is never changed
    assert "(lt.agent_outcome IS NOT NULL\n" in query
    assert "AND lt.agent_outcome IS DISTINCT FROM lt.outcome) AS changed" in query
    # timeouts and failures, counted once if the verdict was stored too
    assert "er.status = 'FAILED'" in query
    assert "NOT EXISTS" in query
    assert "date_bin" not in query  # Postgres 14 only


def test_confidence_bands_are_exact():
    # numeric, not float: 0.7 reads as band 3, where float width_bucket puts it
    query, _ = queries.get_eval_dashboard_query(_filters(), "quality", TEMPLATE_ID)
    assert "(item ->> 'confidence')::numeric" in query
    assert "::float, 0.5, 1.0, 5" not in query


@pytest.mark.parametrize(
    "extra, bound",
    [
        ({}, False),  # a question alone filters nothing
        ({"answer": "CALM"}, True),
        ({"only_unsure": True}, True),
        ({"only_unsure": False}, False),
    ],
)
def test_a_question_is_bound_only_when_a_clause_uses_it(extra, bound):
    query, values = queries.get_eval_calls_query(
        _filters(question_key="tone", **extra), "quality", 50
    )
    # every bound value is referenced: Postgres refuses one it can't place
    assert ("tone" in values) is bound
    placeholders = {int(n) for n in __import__("re").findall(r"\$(\d+)", query)}
    assert placeholders == set(range(1, len(values) + 1))


def test_a_page_of_calls_reads_after_the_cursor():
    started = datetime(2026, 10, 8, 6, tzinfo=timezone.utc)
    query, values = queries.get_outcome_correctness_calls_query(
        _filters(check_result="changed", agent_outcome="BUSY"),
        50,
        started,
        "00000000-0000-0000-0000-000000000009",
    )
    assert "check_result = $" in query and "agent_word = $" in query
    assert "(started_at, id) < (" in query
    assert values[-1] == 51  # one more than the page tells if there is more


# --- the outcome_correctness dashboard -----------------------------------------------


def _row(bucket: str, result: str, checks: int, **extra: Any) -> Dict[str, Any]:
    return {
        "bucket": datetime.fromisoformat(f"2026-10-08T{bucket}"),
        "check_result": result,
        "agent_outcome": extra.get("agent"),
        "eval_outcome": extra.get("eval"),
        "band": extra.get("band"),
        "checks": checks,
        "confidence_sum": extra.get("confidence_sum"),
    }


def test_the_outcome_dashboard_adds_up_every_check():
    rows: List[Dict[str, Any]] = [
        _row("11:45", "agreed", 5),
        _row(
            "11:45",
            "changed",
            2,
            agent="BUSY",
            eval="NOT_INTERESTED",
            band=4,
            confidence_sum=1.7,
        ),
        _row(
            "12:00",
            "changed",
            1,
            agent="BUSY",
            eval="NOT_INTERESTED",
            band=5,
            confidence_sum=0.95,
        ),
        _row(
            "12:00",
            "kept",
            3,
            agent="BUSY",
            eval="USER_BUSY",
            band=3,
            confidence_sum=2.1,
        ),
        _row("12:00", "no_decision", 4, agent="CONFIRM"),
        _row("18:15", "timed_out", 2),
        _row("18:15", "failed", 1),
    ]

    dashboard = analytics.outcome_correctness_dashboard(rows, DAY, DAY)

    assert dashboard["totals"] == {
        "checked": 18,
        "agreed": 5,
        "changed": 3,
        "kept": 3,
        "no_decision": 4,
        "no_answer": 0,
        "timed_out": 2,
        "failed": 1,
    }
    # every 15 minutes of the IST day, empty ones too
    assert dashboard["bucket_minutes"] == 15
    assert len(dashboard["trend"]) == 96
    noon = dashboard["trend"][48]
    assert noon["bucket"] == "2026-10-08T12:00:00+05:30"
    assert (noon["changed"], noon["kept"], noon["no_decision"]) == (1, 3, 4)
    changed, kept = dashboard["pairs"]
    assert (changed["calls"], changed["average_confidence"]) == (3, 0.883)
    assert (kept["check_result"], kept["eval_outcome"]) == ("kept", "USER_BUSY")
    assert dashboard["by_agent_outcome"][0] == {
        "agent_outcome": "BUSY",
        "total": 6,
        "no_decision": 0,
        "kept": 3,
        "changed": 3,
    }
    bands = {band["band"]: band for band in dashboard["confidence"]}
    assert (bands[3]["kept"], bands[4]["changed"], bands[5]["changed"]) == (3, 2, 1)
    assert (bands[4]["from"], bands[4]["to"]) == (0.8, 0.9)


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.setattr(analytics, "_validate_access", AsyncMock())


def _settings(enabled: bool) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "configuration": {"model": "jev-1.13.0", "min_confidence": 0.8},
        "own_enabled": None,
        "own_configuration": None,
    }


async def test_an_agent_with_the_eval_off_sees_only_that(monkeypatch, scope):
    rows = AsyncMock()
    monkeypatch.setattr(analytics, "get_outcome_correctness_dashboard_rows", rows)
    monkeypatch.setattr(
        analytics,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings(False)),
    )

    response = await analytics.get_outcome_correctness_dashboard_analytics(
        _filters(), {}, USER
    )

    assert response["settings"]["enabled"] is False
    assert "totals" not in response
    rows.assert_not_awaited()


async def test_an_agent_with_the_eval_on_gets_its_dashboard(monkeypatch, scope):
    monkeypatch.setattr(
        analytics,
        "get_outcome_correctness_dashboard_rows",
        AsyncMock(return_value=[_row("11:45", "agreed", 1)]),
    )
    monkeypatch.setattr(
        analytics,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings(True)),
    )

    response = await analytics.get_outcome_correctness_dashboard_analytics(
        _filters(), {}, USER
    )

    assert response["settings"]["min_confidence"] == 0.8
    assert response["totals"]["agreed"] == 1


@pytest.mark.parametrize(
    "filters, status",
    [
        ({"date_from": DAY, "date_to": DAY}, 422),  # no agent
        (_filters(date_from=date(2026, 6, 1)), 422),  # past 92 days
        (_filters(date_from=date(2026, 10, 9)), 422),  # backwards
    ],
)
async def test_dashboard_filters_are_checked(monkeypatch, scope, filters, status):
    with pytest.raises(HTTPException) as exc:
        await analytics.get_outcome_correctness_dashboard_analytics(filters, {}, USER)
    assert exc.value.status_code == status


async def test_outcome_calls_page_with_a_cursor(monkeypatch, scope):
    started = datetime(2026, 10, 8, 6, tzinfo=timezone.utc)
    rows = [
        {
            "id": f"00000000-0000-0000-0000-00000000000{n}",
            "source_id": f"lead-{n}",
            "call_id": "CA",
            "started_at": started,
            "check_result": "changed",
            "agent_outcome": "BUSY",
            "eval_outcome": "NOT_INTERESTED",
            "confidence": 0.9,
            "duration_seconds": 42,
            "ended_by": "customer",
        }
        for n in range(1, 4)
    ]
    calls = AsyncMock(return_value=rows)
    monkeypatch.setattr(analytics, "get_outcome_correctness_calls", calls)
    monkeypatch.setattr(
        analytics,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings(True)),
    )

    response = await analytics.get_outcome_correctness_calls_analytics(
        _filters(check_result="changed"), {"limit": 2}, USER
    )

    assert [call["source_id"] for call in response["results"]] == ["lead-1", "lead-2"]
    assert response["pagination"]["has_more"] is True
    cursor = response["pagination"]["next_cursor"]
    assert cursor == f"{started.isoformat()}|{rows[1]['id']}"

    await analytics.get_outcome_correctness_calls_analytics(
        _filters(), {"limit": 2, "cursor": cursor}, USER
    )
    assert calls.await_args_list[-1].args[2:] == (started, rows[1]["id"])


async def test_a_bad_cursor_is_refused(monkeypatch, scope):
    monkeypatch.setattr(
        analytics,
        "get_outcome_correctness_settings",
        AsyncMock(return_value=_settings(True)),
    )
    with pytest.raises(HTTPException) as exc:
        await analytics.get_outcome_correctness_calls_analytics(
            _filters(), {"cursor": "nope"}, USER
        )
    assert exc.value.status_code == 400


# --- custom evals: answers as the dashboard counts them ------------------------------


@pytest.mark.parametrize(
    "result, answer, band",
    [
        ({"kind": "choice", "value": "CALM", "confidence": 0.7}, "CALM", 3),
        ({"kind": "choice", "value": "CALM", "confidence": 0.8}, "CALM", 4),
        ({"kind": "choice", "value": "CALM", "confidence": 1.0}, "CALM", 5),
        ({"kind": "choice", "value": "CALM", "confidence": 0.49}, "CALM", 0),
        ({"kind": "score", "value": 2.5, "confidence": 0.9}, "3", 5),
        ({"kind": "score", "value": 2.49, "confidence": None}, "2", None),
        ({"kind": "noul", "value": 0.5}, "yes", 1),
        ({"kind": "noul", "value": 0.3}, "no", 3),
        ({"kind": "choice", "value": None, "confidence": None}, None, None),
    ],
)
def test_an_answer_and_its_band(result, answer, band):
    assert analytics.answer_of(result) == answer
    assert analytics.band_of(result) == band


def test_a_questions_possible_answers_come_from_its_config():
    choice = {"type": "choice", "criteria": {"CALM": "calm", "UPSET": "upset"}}
    assert analytics.question_answers(choice) == [("CALM", "calm"), ("UPSET", "upset")]
    # a 2-level score is stored 0 to 1, longer ones 1 to N
    assert [
        a
        for a, _ in analytics.question_answers(
            {"type": "score", "criteria": ["x", "y"]}
        )
    ] == ["0", "1"]
    assert analytics.question_answers({"type": "score", "criteria": ["x", "y", "z"]})[
        2
    ] == ("3", "z")
    assert analytics.question_answers({"type": "noul"}) == [("yes", None), ("no", None)]
    # the prompt engine names no answers
    assert analytics.question_answers({"type": "choice"}) == []


CONFIGURATION = {
    "questions": [
        {
            "type": "choice",
            "key": "tone",
            "label": "Tone",
            "criteria": {"CALM": "c", "UPSET": "u"},
        },
        {
            "type": "score",
            "key": "adherence",
            "label": "Adherence",
            "criteria": ["a", "b", "c"],
        },
    ]
}


def _answer(
    key: str, kind: str, answer: Any, count: int, band: Any = None, **extra: Any
):
    return {
        "row_type": "answer",
        "question_key": key,
        "label": key,
        "kind": kind,
        "answer": answer,
        "band": band,
        "count": count,
        "value_sum": extra.get("value_sum"),
        "scale_min": extra.get("scale_min"),
        "scale_max": extra.get("scale_max"),
    }


def test_the_eval_dashboard_follows_the_config():
    rows = [
        {"row_type": "calls", "count": 10},
        {"row_type": "failed", "count": 1},
        _answer("tone", "choice", "UPSET", 4, band=5),
        _answer("tone", "choice", "OLD", 1, band=2),
        _answer("tone", "choice", None, 2),
        _answer(
            "adherence",
            "score",
            "3",
            6,
            band=4,
            value_sum=17.0,
            scale_min=1,
            scale_max=3,
        ),
        _answer(
            "adherence",
            "score",
            "1",
            2,
            band=1,
            value_sum=2.0,
            scale_min=1,
            scale_max=3,
        ),
        _answer("removed", "noul", "yes", 3),
    ]

    dashboard = analytics.eval_dashboard(rows, CONFIGURATION)

    assert dashboard["totals"] == {"calls": 10, "failed": 1}
    tone, adherence, removed = dashboard["questions"]
    # configured answers first, zero counts included, then what the config lost
    assert [(a["answer"], a["count"], a["in_config"]) for a in tone["answers"]] == [
        ("CALM", 0, True),
        ("UPSET", 4, True),
        ("OLD", 1, False),
    ]
    assert (tone["no_answer"], tone["answered"], tone["unsure"]) == (2, 5, 1)
    assert tone["confidence"] == [0, 0, 1, 0, 0, 4]
    assert [
        (a["answer"], a["count"], a["description"]) for a in adherence["answers"]
    ] == [
        ("1", 2, "a"),
        ("2", 0, "b"),
        ("3", 6, "c"),
    ]
    assert adherence["average"] == 2.375
    assert adherence["scale"] == {"min": 1.0, "max": 3.0}
    # a question older results hold but the config no longer asks
    assert (removed["key"], removed["in_config"]) == ("removed", False)
    # no confidence at all: the prompt engine's, or here none recorded
    assert removed["confidence"] is None and removed["unsure"] is None


async def test_a_disabled_eval_shows_nothing(monkeypatch, scope):
    rows = AsyncMock()
    monkeypatch.setattr(analytics, "get_eval_dashboard_rows", rows)
    monkeypatch.setattr(
        analytics,
        "get_custom_eval",
        AsyncMock(return_value={"id": "x", "name": "quality", "enabled": False}),
    )

    with pytest.raises(HTTPException) as exc:
        await analytics.get_eval_dashboard_analytics(
            _filters(eval_name="quality"), {}, USER
        )

    assert exc.value.status_code == 404
    rows.assert_not_awaited()


async def test_an_enabled_eval_has_a_dashboard(monkeypatch, scope):
    rows = AsyncMock(return_value=[{"row_type": "calls", "count": 3}])
    monkeypatch.setattr(analytics, "get_eval_dashboard_rows", rows)
    monkeypatch.setattr(
        analytics,
        "get_custom_eval",
        AsyncMock(
            return_value={
                "id": "eval-id",
                "name": "quality",
                "enabled": True,
                "configuration": CONFIGURATION,
            }
        ),
    )

    response = await analytics.get_eval_dashboard_analytics(
        _filters(eval_name="quality"), {}, USER
    )

    rows.assert_awaited_once_with(_filters(eval_name="quality"), "quality", "eval-id")
    assert response["totals"]["calls"] == 3
    assert [q["key"] for q in response["questions"]] == ["tone", "adherence"]


async def test_a_custom_eval_needs_its_name(scope):
    with pytest.raises(HTTPException) as exc:
        await analytics.get_eval_dashboard_analytics(_filters(), {}, USER)
    assert exc.value.status_code == 422


async def test_eval_calls_carry_each_answer_as_counted(monkeypatch, scope):
    monkeypatch.setattr(
        analytics,
        "get_custom_eval",
        AsyncMock(return_value={"id": "x", "name": "quality", "enabled": True}),
    )
    monkeypatch.setattr(
        analytics,
        "get_eval_calls",
        AsyncMock(
            return_value=[
                {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "source_id": "lead-1",
                    "started_at": datetime(2026, 10, 8, tzinfo=timezone.utc),
                    "answers": [
                        {
                            "key": "tone",
                            "label": "Tone",
                            "kind": "choice",
                            "value": "CALM",
                            "confidence": 0.7,
                        },
                        {
                            "key": "offer",
                            "label": "Offer",
                            "kind": "noul",
                            "value": 0.95,
                        },
                    ],
                }
            ]
        ),
    )

    response = await analytics.get_eval_calls_analytics(
        _filters(eval_name="quality"), {}, USER
    )

    tone, offer = response["results"][0]["answers"]
    assert (tone["answer"], tone["band"], tone["unsure"]) == ("CALM", 3, True)
    assert (offer["answer"], offer["band"], offer["unsure"]) == ("yes", 5, False)


def test_every_eval_analytics_type_is_served():
    for kind in (
        AnalyticsType.OUTCOME_CORRECTNESS_DASHBOARD,
        AnalyticsType.OUTCOME_CORRECTNESS_CALLS,
        AnalyticsType.EVAL_DASHBOARD,
        AnalyticsType.EVAL_CALLS,
    ):
        assert kind in _ANALYTICS_HANDLERS
