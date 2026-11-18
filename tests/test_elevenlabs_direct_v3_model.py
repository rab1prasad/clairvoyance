"""A direct call to ElevenLabs never asks for a v3 model.

ElevenLabs refuses every v3 name on the endpoints the app dials itself (the
streaming one with HTTP 400, which left the bot silent for the whole call), so
a template on a v3 voice that is not going through DragonTTS speaks on
eleven_flash_v2_5 with the same voice id.
"""

import pytest

from app.ai.voice.tts import elevenlabs
from app.ai.voice.tts.elevenlabs import (
    ElevenLabsConfig,
    _generate_elevenlabs_audio,
    build_elevenlabs_tts,
)

V3_MODELS = [
    "eleven_v3",
    "eleven_v3_conversational",
    "eleven_v3_conversational_clean_tempo",
    "eleven_v3_conversational_clean_tempo_v2",
]
OTHER_MODELS = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"]


def _streaming_address(model: str) -> str:
    tts = build_elevenlabs_tts(
        ElevenLabsConfig(
            api_key="test",
            url="wss://api.in.residency.elevenlabs.io",
            voice_id="voice",
            model=model,
        )
    )
    return tts._build_websocket_url()


@pytest.mark.parametrize("model", V3_MODELS)
def test_streaming_call_asks_for_flash_instead_of_a_v3_model(model):
    address = _streaming_address(model)

    assert "model_id=eleven_flash_v2_5&" in address
    assert "/v1/text-to-speech/voice/multi-stream-input" in address


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_streaming_call_keeps_any_other_model(model):
    assert f"model_id={model}&" in _streaming_address(model)


class _Response:
    content = b"audio"

    def raise_for_status(self):
        pass


class _Client:
    """Stands in for httpx.AsyncClient: records the one POST and answers it."""

    posts: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        _Client.posts.append({"url": url, **kwargs})
        return _Response()


async def _one_shot_model(monkeypatch, model):
    _Client.posts = []
    monkeypatch.setattr(elevenlabs.httpx, "AsyncClient", _Client)

    audio = await _generate_elevenlabs_audio(
        text="ठीक है",
        voice_id="voice",
        model_id=model,
        api_key="test",
        base_url="https://api.in.residency.elevenlabs.io",
    )

    assert audio == b"audio"
    assert len(_Client.posts) == 1
    assert "/v1/text-to-speech/voice?" in _Client.posts[0]["url"]
    return _Client.posts[0]["json"]["model_id"]


@pytest.mark.parametrize("model", V3_MODELS)
async def test_one_shot_call_asks_for_flash_instead_of_a_v3_model(monkeypatch, model):
    assert await _one_shot_model(monkeypatch, model) == "eleven_flash_v2_5"


@pytest.mark.parametrize("model", OTHER_MODELS)
async def test_one_shot_call_keeps_any_other_model(monkeypatch, model):
    assert await _one_shot_model(monkeypatch, model) == model
