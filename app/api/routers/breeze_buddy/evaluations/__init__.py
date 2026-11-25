from fastapi import APIRouter, Depends, Query, status

from app.api.security.breeze_buddy.rbac_token import get_current_user_with_rbac
from app.schemas import UserInfo
from app.schemas.breeze_buddy.evals import (
    CreateCustomEvalRequest,
    CustomEvalListResponse,
    CustomEvalResponse,
    EvaluationConfigResponse,
    EvaluationEnableRequest,
    EvaluationType,
    OutcomeCorrectnessResponse,
    SaveEvaluationConfigurationRequest,
    SetCustomEvalEnabledRequest,
    UpdateCustomEvalRequest,
    UpdateOutcomeCorrectnessRequest,
)

from .custom.handlers import (
    create_custom_eval_handler,
    get_custom_eval_handler,
    list_custom_evals_handler,
    set_custom_eval_enabled_handler,
    update_custom_eval_handler,
)
from .handlers import (
    get_evaluation_config_handler,
    save_evaluation_configuration_handler,
    set_evaluation_enabled_handler,
)
from .preset.outcome_correctness import (
    get_outcome_correctness_handler,
    update_outcome_correctness_handler,
)

router = APIRouter()


@router.get(
    "/templates/{template_id}/evaluations",
    response_model=EvaluationConfigResponse,
)
async def get_evaluation_config(
    template_id: str,
    evaluation_type: EvaluationType = Query(...),
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await get_evaluation_config_handler(
        template_id, evaluation_type, current_user
    )


@router.patch(
    "/templates/{template_id}/evaluations/config",
    response_model=EvaluationConfigResponse,
)
async def set_evaluation_enabled(
    template_id: str,
    request: EvaluationEnableRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await set_evaluation_enabled_handler(template_id, request, current_user)


@router.post(
    "/templates/{template_id}/evaluations/configuration",
    response_model=EvaluationConfigResponse,
)
async def save_evaluation_configuration(
    template_id: str,
    request: SaveEvaluationConfigurationRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await save_evaluation_configuration_handler(
        template_id, request, current_user
    )


# --- the preset outcome_correctness eval ------------------------------------
# Anyone with access to the agent may turn it on or off; the threshold is
# an edit, admins only. template_id 'default' addresses the platform default
# (admins only).


@router.get(
    "/templates/{template_id}/evaluations/outcome-correctness",
    response_model=OutcomeCorrectnessResponse,
)
async def get_outcome_correctness(
    template_id: str,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await get_outcome_correctness_handler(template_id, current_user)


@router.patch(
    "/templates/{template_id}/evaluations/outcome-correctness",
    response_model=OutcomeCorrectnessResponse,
)
async def update_outcome_correctness(
    template_id: str,
    request: UpdateOutcomeCorrectnessRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await update_outcome_correctness_handler(template_id, request, current_user)


# --- the agent's custom evals -------------------------------------------------
# Anyone with access to the agent may list, create and turn one on or off
# (PATCH enabled); replacing a configuration (PUT) is an edit, admins only.
# No delete: an eval is turned off instead.


@router.get(
    "/templates/{template_id}/evaluations/custom",
    response_model=CustomEvalListResponse,
)
async def list_custom_evals(
    template_id: str,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await list_custom_evals_handler(template_id, current_user)


@router.post(
    "/templates/{template_id}/evaluations/custom",
    response_model=CustomEvalResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_custom_eval(
    template_id: str,
    request: CreateCustomEvalRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await create_custom_eval_handler(template_id, request, current_user)


@router.get(
    "/templates/{template_id}/evaluations/custom/{name}",
    response_model=CustomEvalResponse,
)
async def get_custom_eval(
    template_id: str,
    name: str,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await get_custom_eval_handler(template_id, name, current_user)


@router.put(
    "/templates/{template_id}/evaluations/custom/{name}",
    response_model=CustomEvalResponse,
)
async def update_custom_eval(
    template_id: str,
    name: str,
    request: UpdateCustomEvalRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await update_custom_eval_handler(template_id, name, request, current_user)


@router.patch(
    "/templates/{template_id}/evaluations/custom/{name}",
    response_model=CustomEvalResponse,
)
async def set_custom_eval_enabled(
    template_id: str,
    name: str,
    request: SetCustomEvalEnabledRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    return await set_custom_eval_enabled_handler(
        template_id, name, request, current_user
    )
