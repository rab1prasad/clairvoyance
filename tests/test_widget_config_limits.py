"""A widget's hourly limits cannot be set to 0, and no limit goes past the
integer column it is stored in.

The rate limiter reads a limit of 0 as "limit off" (an ops switch). The
widget config API accepted 0, so one request could remove the per-IP cap on
a widget's anonymous sessions, messages or voice sessions.

The four limit columns are Postgres integers. A value above 2147483647
passed validation and failed the write with a 500.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.breeze_buddy.widget_config import (
    PG_INTEGER_MAX,
    WidgetConfigCreate,
    WidgetConfigUpdate,
)

HOURLY_LIMITS = [
    "max_sessions_per_ip_hour",
    "max_messages_per_ip_hour",
    "max_voice_sessions_per_ip_hour",
]
ALL_LIMITS = HOURLY_LIMITS + ["max_concurrent_per_ip"]
CREATE = {"reseller_id": "R1", "merchant_id": "acme-shop", "template_id": "t1"}


@pytest.mark.parametrize("field", HOURLY_LIMITS)
@pytest.mark.parametrize("value", [0, -1])
def test_create_refuses_a_limit_below_one(field: str, value: int) -> None:
    with pytest.raises(ValidationError) as refused:
        WidgetConfigCreate.model_validate({**CREATE, field: value})
    assert [error["loc"] for error in refused.value.errors()] == [(field,)]


@pytest.mark.parametrize("field", HOURLY_LIMITS)
@pytest.mark.parametrize("value", [0, -1])
def test_update_refuses_a_limit_below_one(field: str, value: int) -> None:
    with pytest.raises(ValidationError) as refused:
        WidgetConfigUpdate.model_validate({field: value})
    assert [error["loc"] for error in refused.value.errors()] == [(field,)]


@pytest.mark.parametrize("field", HOURLY_LIMITS)
def test_one_is_the_least_a_limit_can_be(field: str) -> None:
    assert getattr(WidgetConfigCreate.model_validate({**CREATE, field: 1}), field) == 1
    assert getattr(WidgetConfigUpdate.model_validate({field: 1}), field) == 1


def test_the_bound_is_the_postgres_integer_maximum() -> None:
    assert PG_INTEGER_MAX == 2**31 - 1


@pytest.mark.parametrize("field", ALL_LIMITS)
def test_a_limit_past_the_integer_column_is_refused(field: str) -> None:
    too_big = PG_INTEGER_MAX + 1
    with pytest.raises(ValidationError) as refused:
        WidgetConfigCreate.model_validate({**CREATE, field: too_big})
    assert [error["loc"] for error in refused.value.errors()] == [(field,)]
    with pytest.raises(ValidationError) as refused:
        WidgetConfigUpdate.model_validate({field: too_big})
    assert [error["loc"] for error in refused.value.errors()] == [(field,)]


@pytest.mark.parametrize("field", ALL_LIMITS)
def test_the_integer_column_maximum_is_accepted(field: str) -> None:
    created = WidgetConfigCreate.model_validate({**CREATE, field: PG_INTEGER_MAX})
    assert getattr(created, field) == PG_INTEGER_MAX
    updated = WidgetConfigUpdate.model_validate({field: PG_INTEGER_MAX})
    assert getattr(updated, field) == PG_INTEGER_MAX
