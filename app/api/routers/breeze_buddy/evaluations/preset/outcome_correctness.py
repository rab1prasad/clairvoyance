"""The preset outcome_correctness eval's settings.

For an agent: anyone with access to it can turn the eval on or off; changing
the confidence its answer needs before it replaces the agent's outcome is an
edit, for admins only. For template 'default' (admins only): the platform
default every agent without a setting of its own follows, model included —
every agent runs the preset's model.

An agent's own row decides its flag once it has one (a new row starts on the
default's flag). Its min_confidence overrides the default's only while set;
sending it as null puts the agent back on the default.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException, status

from app.core.security.authorization import require_admin
from app.database.accessor.breeze_buddy.evaluation_config import (
    get_outcome_correctness_settings,
    save_outcome_correctness_setting,
    update_outcome_correctness_default,
)
from app.schemas import UserInfo
from app.schemas.breeze_buddy.evals import (
    OutcomeCorrectnessDefaults,
    OutcomeCorrectnessResponse,
    UpdateOutcomeCorrectnessRequest,
)
from app.utils.common import parse_json

from ..handlers import _validate_access

#: The path's template id that addresses the platform default.
DEFAULT_TEMPLATE = "default"

_MIN_CONFIDENCE = "min_confidence"
_MODEL = "model"


def outcome_correctness_view(
    row: Dict[str, Any], template_id: Optional[str]
) -> OutcomeCorrectnessResponse:
    """PURE: the settings as they run, from the preset row and the agent's
    own (``own_*`` NULL without one)."""
    preset = parse_json(row, "configuration") or {}
    own = parse_json(row, "own_configuration") or {}
    defaults = OutcomeCorrectnessDefaults(
        enabled=bool(row["enabled"]),
        min_confidence=float(preset[_MIN_CONFIDENCE]),
    )
    own_enabled = row.get("own_enabled")
    return OutcomeCorrectnessResponse(
        template_id=UUID(str(template_id)) if template_id else None,
        enabled=defaults.enabled if own_enabled is None else bool(own_enabled),
        min_confidence=float(own.get(_MIN_CONFIDENCE, defaults.min_confidence)),
        model=str(preset.get(_MODEL) or ""),
        defaults=defaults,
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="The outcome_correctness preset is not set up",
    )


async def _settings(template_id: Optional[str]) -> OutcomeCorrectnessResponse:
    row = await get_outcome_correctness_settings(template_id)
    if not row:
        raise _not_found()
    return outcome_correctness_view(row, template_id)


async def get_outcome_correctness_handler(
    template_id: str,
    current_user: UserInfo,
) -> OutcomeCorrectnessResponse:
    if template_id == DEFAULT_TEMPLATE:
        require_admin(current_user)
        return await _settings(None)
    await _validate_access(template_id, current_user)
    return await _settings(template_id)


async def update_outcome_correctness_handler(
    template_id: str,
    request: UpdateOutcomeCorrectnessRequest,
    current_user: UserInfo,
) -> OutcomeCorrectnessResponse:
    sent = request.model_fields_set
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Nothing to change"
        )
    if "enabled" in sent and request.enabled is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="enabled must be true or false",
        )

    if template_id == DEFAULT_TEMPLATE:
        require_admin(current_user)
        if (_MIN_CONFIDENCE in sent and request.min_confidence is None) or (
            _MODEL in sent and request.model is None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The default needs a min_confidence and a model",
            )
        settings: Dict[str, Any] = {
            key: getattr(request, key)
            for key in (_MIN_CONFIDENCE, _MODEL)
            if key in sent
        }
        if not await update_outcome_correctness_default(request.enabled, settings):
            raise _not_found()
        return await _settings(None)

    await _validate_access(template_id, current_user)
    if _MIN_CONFIDENCE in sent:
        # an edit: admins only (anyone with access may turn the eval on/off)
        require_admin(current_user)
    if _MODEL in sent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The model is the platform default's: every agent runs it",
        )
    settings = {}
    cleared: List[str] = []
    if _MIN_CONFIDENCE in sent:
        if request.min_confidence is None:
            cleared.append(_MIN_CONFIDENCE)
        else:
            settings[_MIN_CONFIDENCE] = request.min_confidence
    if not await save_outcome_correctness_setting(
        str(template_id), request.enabled, settings, cleared
    ):
        raise _not_found()
    return await _settings(template_id)
