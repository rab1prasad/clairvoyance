"""The agent's custom evals.

Anyone with access to the agent can list and read them, create one, and turn
one on or off. Editing one (replacing its configuration) is for admins only.

Each is one of the agent's CONVERSATION_EVALS rows, addressed by its name.
The name is fixed once made: the eval's results are stored under it. No eval
is ever deleted; it is turned off instead, and a disabled eval neither runs
nor shows in analytics. Only structured (Jev) evals run for now
(app/services/evals/custom/batch.py), so only they can be made here.
"""

import copy
from typing import Any, Dict
from uuid import UUID

from fastapi import HTTPException, status

from app.core.security.authorization import require_admin
from app.database.accessor.breeze_buddy.evaluation_config import (
    create_custom_eval,
    get_custom_eval,
    list_custom_evals,
    set_custom_eval_enabled,
    update_custom_eval_configuration,
)
from app.database.queries.breeze_buddy.evaluation_config import NOT_CUSTOM_EVALS
from app.schemas import UserInfo
from app.schemas.breeze_buddy.evals import (
    CreateCustomEvalRequest,
    CustomEvalListResponse,
    CustomEvalResponse,
    EvaluationType,
    SetCustomEvalEnabledRequest,
    UpdateCustomEvalRequest,
)
from app.services.evals.definition import validate_evals_configuration
from app.services.evals.engines.structured_judge_engine import StructuredJudgeEngine
from app.utils.common import parse_json

from ..handlers import _validate_access

#: Names a custom eval cannot take: the TOPIC row's, and the preset's
#: (NOT_CUSTOM_EVALS), which the custom-eval queries never address either.
RESERVED_NAMES = frozenset({EvaluationType.TOPIC.value.lower(), *NOT_CUSTOM_EVALS})


def _validated(configuration: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration as stored (the merchant's own JSON), once its
    engine accepts it. Raises ``ValueError``."""
    validate_evals_configuration(configuration)
    if configuration.get("engine") != StructuredJudgeEngine.name:
        raise ValueError(
            f"only {StructuredJudgeEngine.name!r} evals run for now; "
            f"got {configuration.get('engine')!r}"
        )
    return copy.deepcopy(configuration)


def _invalid(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid eval configuration: {exc}",
    )


def _not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No eval named {name!r} for this agent",
    )


def _response(row: Dict[str, Any]) -> CustomEvalResponse:
    return CustomEvalResponse(
        id=row["id"],
        template_id=row["template_id"],
        name=row["name"],
        enabled=bool(row.get("enabled")),
        # asyncpg hands jsonb back as text; parse_json takes either form
        configuration=parse_json(row, "configuration") or {},
    )


async def list_custom_evals_handler(
    template_id: str,
    current_user: UserInfo,
) -> CustomEvalListResponse:
    await _validate_access(template_id, current_user)
    rows = await list_custom_evals(str(template_id))
    return CustomEvalListResponse(
        template_id=UUID(str(template_id)),
        evals=[_response(row) for row in rows],
    )


async def create_custom_eval_handler(
    template_id: str,
    request: CreateCustomEvalRequest,
    current_user: UserInfo,
) -> CustomEvalResponse:
    await _validate_access(template_id, current_user)
    if request.name in RESERVED_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{request.name!r} is reserved; pick another name",
        )
    try:
        configuration = _validated(request.configuration)
    except (TypeError, ValueError) as exc:
        raise _invalid(exc) from exc
    row = await create_custom_eval(
        str(template_id), request.name, configuration, request.enabled
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This agent already has an eval named {request.name!r}",
        )
    return _response(row)


async def get_custom_eval_handler(
    template_id: str,
    name: str,
    current_user: UserInfo,
) -> CustomEvalResponse:
    await _validate_access(template_id, current_user)
    row = await get_custom_eval(str(template_id), name)
    if not row:
        raise _not_found(name)
    return _response(row)


async def update_custom_eval_handler(
    template_id: str,
    name: str,
    request: UpdateCustomEvalRequest,
    current_user: UserInfo,
) -> CustomEvalResponse:
    # an edit: admins only (anyone with access may create one or turn it on/off)
    require_admin(current_user)
    await _validate_access(template_id, current_user)
    try:
        configuration = _validated(request.configuration)
    except (TypeError, ValueError) as exc:
        raise _invalid(exc) from exc
    row = await update_custom_eval_configuration(str(template_id), name, configuration)
    if not row:
        raise _not_found(name)
    return _response(row)


async def set_custom_eval_enabled_handler(
    template_id: str,
    name: str,
    request: SetCustomEvalEnabledRequest,
    current_user: UserInfo,
) -> CustomEvalResponse:
    await _validate_access(template_id, current_user)
    row = await set_custom_eval_enabled(str(template_id), name, request.enabled)
    if not row:
        raise _not_found(name)
    return _response(row)
