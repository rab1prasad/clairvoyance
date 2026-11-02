"""Transport configuration for voice agents.

Note: VAD (Voice Activity Detection) is configured in the LLMUserAggregator
(via UserTurnStrategies), NOT in the transport. The transport only handles
audio I/O, sample rates, and optional audio filters/mixers.
"""

from typing import Any, Optional

from fastapi import WebSocket
from pipecat.runner.types import CallData
from pipecat.runner.utils import _create_telephony_transport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from app.ai.voice.agents.breeze_buddy.agent.noise_filter import (
    create_audio_input_filter,
)
from app.ai.voice.agents.breeze_buddy.agent.utils import (
    TRANSPORT_TYPE_DAILY,
    TRANSPORT_TYPE_TELEPHONY,
    TRANSPORT_TYPE_VOBIZ,
)
from app.ai.voice.agents.breeze_buddy.services.telephony.vobiz.serializer import (
    VobizFrameSerializer,
)
from app.ai.voice.agents.breeze_buddy.template.types import (
    ConfigurationModel,
    TemplateModel,
)
from app.ai.voice.agents.breeze_buddy.template.vad import TELEPHONY_SAMPLE_RATE
from app.ai.voice.agents.breeze_buddy.utils.audio_mixer import (
    create_background_sound_mixer,
)
from app.core.config import static


def get_transport_params(
    template: Optional[TemplateModel] = None,
    configurations: Optional[ConfigurationModel] = None,
    daily_audio_out_10ms_chunks: int = 10,
) -> dict:
    """Get transport parameters dictionary for all transport types.

    VAD is not configured here — it lives in the LLMUserAggregator
    (via vad_analyzer + UserTurnStrategies) where it feeds turn detection.

    Args:
        template: Optional template model for background sound mixer configuration
        configurations: Optional configuration model for settings (e.g., noise filter)
        daily_audio_out_10ms_chunks: Daily output write size in 10ms chunks.
            The live default comes from the BB_DAILY_AUDIO_OUT_10MS_CHUNKS
            dynamic config, resolved by the async caller and passed in (this
            function is sync); only the daily transport uses it, so telephony
            callers leave the fallback default.

    Returns:
        Dictionary mapping transport types to parameter factory functions
    """
    # Create separate mixers per transport so each uses the correct sample rate.
    # Daily runs at 24 kHz; all telephony transports run at TELEPHONY_SAMPLE_RATE (8 kHz).
    daily_mixer = create_background_sound_mixer(template, sample_rate=24000)
    telephony_mixer = create_background_sound_mixer(
        template, sample_rate=TELEPHONY_SAMPLE_RATE
    )

    return {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Output write size in 10ms chunks (pipecat default 4 = 40ms):
            # bigger chunks give the paced audio writer more event-loop slack
            # against stalls in the bot process, which otherwise underrun
            # Daily's virtual mic and the browser hears crackling
            # (pipecat#331). Rationale + trade-offs on the
            # BB_DAILY_AUDIO_OUT_10MS_CHUNKS dynamic config.
            audio_out_10ms_chunks=daily_audio_out_10ms_chunks,
            audio_out_mixer=daily_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_DAILY
            ),
        ),
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_mixer=telephony_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_TELEPHONY
            ),
        ),
        "exotel": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_mixer=telephony_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_TELEPHONY
            ),
        ),
        "telnyx": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_mixer=telephony_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_TELEPHONY
            ),
        ),
        "plivo": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_mixer=telephony_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_TELEPHONY
            ),
        ),
        "vobiz": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_mixer=telephony_mixer,
            audio_in_filter=create_audio_input_filter(
                configurations, TRANSPORT_TYPE_TELEPHONY
            ),
        ),
    }


async def create_telephony_transport(
    websocket: WebSocket,
    params: Any,
    transport_type: str,
    call_data: CallData,
) -> BaseTransport:
    """Build the telephony transport for an already-parsed call.

    pipecat's factory knows twilio/telnyx/plivo/exotel. A Vobiz stream opens
    with a Plivo-shaped start event, so pipecat detects it as "plivo" and
    would hang up through api.plivo.com with PLIVO_* keys. The agent relabels
    those calls "vobiz" from the websocket path, and they get
    VobizFrameSerializer here instead. Used by the first build and by the
    agent-to-agent rebuild, so both stay on the same serializer.
    """
    if transport_type != TRANSPORT_TYPE_VOBIZ:
        return await _create_telephony_transport(
            websocket, params, transport_type, call_data
        )

    params.add_wav_header = False
    params.serializer = VobizFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=static.VOBIZ_AUTH_ID,
        auth_token=static.VOBIZ_AUTH_TOKEN,
    )
    return FastAPIWebsocketTransport(websocket=websocket, params=params)
