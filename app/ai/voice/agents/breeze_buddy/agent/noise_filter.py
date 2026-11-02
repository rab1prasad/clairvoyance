"""Noise filters on the caller's audio (the transport's ``audio_in_filter``).

The template's ``noise_filter`` block alone decides the filter: Noizzen
(in-house, app/ai/voice/noise_filter/noizzen) or ai-coustics (AIC). Every
failure here proceeds without a filter, so a missing model never aborts a call.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from aic_sdk import Model as AICModel
from pipecat.audio.filters.aic_filter import AICFilter
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter

from app.ai.voice.agents.breeze_buddy.agent.utils import TRANSPORT_TYPE_DAILY
from app.ai.voice.agents.breeze_buddy.template.types import (
    ConfigurationModel,
    NoiseFilterModel,
    NoiseFilterProvider,
    NoiseFilterType,
)
from app.ai.voice.noise_filter.noizzen import NoizzenFilter, find_model_file
from app.core.config import static
from app.core.logger import logger

_missing_models_logged: set = set()


def _create_noizzen_filter(
    model: NoiseFilterModel,
    transport_type: str,
    enhancement_level: Optional[float],
) -> Optional[BaseAudioFilter]:
    """The Noizzen filter for a template model, or None when its file is missing."""
    name = (
        "dpdfnet_base_16k" if transport_type == TRANSPORT_TYPE_DAILY else "dpdfnet2_8k"
    )
    path = find_model_file(name, Path(static.BREEZE_BUDDY_NOISE_MODEL_DIR))
    if path is None:
        if name not in _missing_models_logged:  # once per process, not per call
            _missing_models_logged.add(name)
            logger.warning(
                "Noise model {} is not provisioned in {}; proceeding without a "
                "noise filter",
                name,
                static.BREEZE_BUDDY_NOISE_MODEL_DIR,
            )
        return None
    try:
        noise_filter = NoizzenFilter(
            model=name, model_path=path, enhancement_level=enhancement_level
        )
    except Exception as e:
        logger.warning(f"Failed to create noise filter, proceeding without it: {e}")
        return None
    logger.info(
        "Noise filter initialized: provider=noizzen, model={} ({}), "
        "enhancement_level={}",
        model.value,
        name,
        enhancement_level,
    )
    return noise_filter


def _aic_model_path(model: NoiseFilterModel, transport_type: str) -> Path:
    """The artifact for a model on a transport.

    The model value carries the size (noise_cancellation = quail-L,
    noise_cancellation_s = quail-S); the transport picks the processing rate:
    Daily (web) uses the 16kHz artifact, telephony (Twilio/Plivo/Exotel) the 8kHz
    one. These rates describe the model, not the transport's native rate: Daily
    may run at 24kHz elsewhere in the pipeline while still using the 16kHz model
    here. Voice Focus ships only a 16kHz artifact; the SDK resamples 8kHz input.
    """
    web = transport_type == TRANSPORT_TYPE_DAILY
    paths = {
        NoiseFilterModel.NOISE_CANCELLATION: (
            static.AIC_MODEL_PATH_16KHZ if web else static.AIC_MODEL_PATH
        ),
        NoiseFilterModel.NOISE_CANCELLATION_S: (
            static.AIC_MODEL_PATH_S_16KHZ if web else static.AIC_MODEL_PATH_S
        ),
        NoiseFilterModel.VOICE_FOCUS: static.AIC_VOICE_FOCUS_MODEL_PATH,
    }
    return Path(paths[model])


def _resolve_aic_model(
    model: NoiseFilterModel, transport_type: str
) -> tuple[NoiseFilterModel, Path]:
    """The model to load and its artifact.

    A missing quail-S file falls back to quail-L (noise_cancellation), the model
    every template ran before quail-S existed, so a soft-failed Docker copy
    degrades to the known model instead of dropping the filter. The returned
    model is the one loaded, so logs never report quail-S for a quail-L file.
    """
    path = _aic_model_path(model, transport_type)
    if model == NoiseFilterModel.NOISE_CANCELLATION_S and not path.is_file():
        logger.warning(
            "AIC quail-S model file is unavailable; using noise_cancellation "
            "(quail-L) instead: {}",
            path,
        )
        model = NoiseFilterModel.NOISE_CANCELLATION
        path = _aic_model_path(model, transport_type)
    return model, path


@lru_cache(maxsize=None)
def _aic_model_error(path: Path) -> Optional[str]:
    try:
        AICModel.from_file(path)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def create_audio_input_filter(
    configurations: Optional[ConfigurationModel] = None,
    transport_type: str = TRANSPORT_TYPE_DAILY,
) -> Optional[BaseAudioFilter]:
    """Create audio input filter based on configuration.

    Currently supports:
        - Noizzen (in-house noise suppression, app/ai/voice/noise_filter/noizzen)
        - AIC filter (ai-coustics noise enhancement, preserves all speakers)
        - AIC Voice Focus (foreground/primary-speaker isolation)

    Future filters can be added here with their own enable flags.

    Args:
        configurations: The configuration model containing noise filter settings.
        transport_type: The transport type to determine model selection.

    Returns:
        Audio filter instance if enabled and successfully created, None otherwise.
    """
    # Check if noise filter is configured and enabled
    if not configurations:
        return None

    noise_filter_config = configurations.noise_filter
    if not noise_filter_config or not noise_filter_config.enable:
        return None

    # Resolve either the new provider/model shape or the legacy type shape at
    # runtime without rewriting the template configuration.
    provider = noise_filter_config.provider or NoiseFilterProvider.AIC
    model = noise_filter_config.model or (
        NoiseFilterModel.VOICE_FOCUS
        if noise_filter_config.type == NoiseFilterType.AIC_VOICE_FOCUS
        else NoiseFilterModel.NOISE_CANCELLATION
    )

    if provider == NoiseFilterProvider.NOIZZEN:
        return _create_noizzen_filter(
            model, transport_type, noise_filter_config.enhancement_level
        )

    if provider == NoiseFilterProvider.AIC:
        if not static.BREEZE_BUDDY_AIC_LICENSE_KEY:
            logger.warning("AIC filter enabled but license key not configured")
            return None

        # Docker provisions the models before startup; the application never
        # downloads model artifacts at runtime.
        model, model_path = _resolve_aic_model(model, transport_type)
        if not model_path.is_file():
            # AICFilter validates the artifact only when the transport starts.
            # Fail open here so a soft-failed Docker copy cannot abort a call.
            logger.warning(
                "AIC filter model file is unavailable; proceeding without it: {}",
                model_path,
            )
            return None

        model_error = _aic_model_error(model_path)
        if model_error:
            logger.warning(
                "AIC filter model file does not load; proceeding without it: {} ({})",
                model_path,
                model_error,
            )
            return None

        try:
            if noise_filter_config.enhancement_level is None:
                # Preserve the existing AIC behavior exactly: omitting this
                # argument lets the selected model use its own default.
                aic_filter = AICFilter(
                    license_key=static.BREEZE_BUDDY_AIC_LICENSE_KEY,
                    model_path=model_path,
                )
            else:
                aic_filter = AICFilter(
                    license_key=static.BREEZE_BUDDY_AIC_LICENSE_KEY,
                    model_path=model_path,
                    enhancement_level=noise_filter_config.enhancement_level,
                )
            logger.info(
                "AIC filter initialized: provider={}, model={}, path={}, "
                "enhancement_level={}",
                provider.value,
                model.value,
                model_path,
                noise_filter_config.enhancement_level,
            )
            return aic_filter
        except Exception as e:
            logger.warning(
                f"Failed to initialize AIC filter, proceeding without it: {e}"
            )

    return None
