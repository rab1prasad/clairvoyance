"""The evaluation vocabulary and the shapes of the evaluations API
(/templates/{id}/evaluations*): ``EvaluationType`` names what an
evaluation_config row is, one surface serves every type; the preset
outcome_correctness eval and the agent's custom evals have their own
routes beside it."""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

#: An eval's name, the evaluation_config_name_format_check pattern: what its
#: results are stored under (evaluation_result.result), so it never changes.
EVAL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class EvaluationType(str, Enum):
    TOPIC = "TOPIC"
    CONVERSATION_EVALS = "CONVERSATION_EVALS"


class EvaluationEnableRequest(BaseModel):
    evaluation_type: EvaluationType
    enabled: bool


class EvaluationConfigResponse(BaseModel):
    template_id: UUID
    evaluation_type: EvaluationType
    enabled: bool
    # admin-only; None for other roles
    configuration: Optional[Dict[str, Any]] = None
    # the TOPIC catalog (a separate column); empty for other types
    topics: List[str] = Field(default_factory=list)


class SaveEvaluationConfigurationRequest(BaseModel):
    """The full configuration; the handler validates it via the type's own
    validator, then creates the agent's row (disabled) or replaces the
    configuration of the existing one (CONVERSATION_EVALS only for now)."""

    evaluation_type: EvaluationType
    configuration: Dict[str, Any]


# --- the preset outcome_correctness eval -------------------------------------


class OutcomeCorrectnessDefaults(BaseModel):
    """The platform default every agent without a setting of its own follows."""

    enabled: bool
    min_confidence: float


class OutcomeCorrectnessResponse(BaseModel):
    """The outcome_correctness eval as it runs for one agent, or for template
    'default' the platform default itself (``template_id`` None)."""

    template_id: Optional[UUID] = None
    enabled: bool
    # the eval's answer replaces the agent's word only at or above this
    min_confidence: float
    model: str
    defaults: OutcomeCorrectnessDefaults


class UpdateOutcomeCorrectnessRequest(BaseModel):
    """Only the fields sent change. For an agent, anyone with access may send
    ``enabled``; ``min_confidence`` is an edit, admins only, and sent as null
    puts the agent back on the platform default. ``model`` is the platform
    default's alone (every agent runs the preset's model); the default is
    admins only."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = None
    min_confidence: Optional[float] = Field(None, gt=0, le=1)
    model: Optional[str] = Field(None, min_length=1, max_length=200)


# --- the agent's custom evals ------------------------------------------------


class CustomEvalResponse(BaseModel):
    id: UUID
    template_id: UUID
    name: str
    enabled: bool
    configuration: Dict[str, Any]


class CustomEvalListResponse(BaseModel):
    template_id: UUID
    evals: List[CustomEvalResponse] = Field(default_factory=list)


class CreateCustomEvalRequest(BaseModel):
    """A new eval: its name is fixed once created (its results are stored
    under it); the configuration is validated by its engine. Created
    disabled unless ``enabled`` says otherwise."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=EVAL_NAME_PATTERN)
    configuration: Dict[str, Any]
    enabled: bool = False


class UpdateCustomEvalRequest(BaseModel):
    """The full configuration, replacing the stored one; the enabled flag
    is kept. An edit: admins only."""

    model_config = ConfigDict(extra="forbid")

    configuration: Dict[str, Any]


class SetCustomEvalEnabledRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
