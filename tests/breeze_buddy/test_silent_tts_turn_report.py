"""pipecat's "one TTS turn had no audio" note is not a call error.

pipecat 1.14 pushes an error for every TTS turn that ends without audio. That
includes a sentence whose first audio only arrives after pipecat's 3 s idle
wait and is then spoken anyway. 1.1.0 reported nothing there, so the note is
neither stored on the lead nor shown to the client as a pipeline error. Any
other pipeline error still is.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.ai.voice.agents.breeze_buddy.agent import Agent
from app.ai.voice.agents.breeze_buddy.utils.common import is_silent_tts_turn_report
from app.ai.voice.tts.dragontts import DragonTTSConfig, build_dragontts_tts


async def _pipecat_reports(
    monkeypatch, max_in_a_row: int, silent_turns: int
) -> list[str]:
    """What pipecat itself reports for ``silent_turns`` silent TTS turns in a row."""
    tts = build_dragontts_tts(
        DragonTTSConfig(
            url="https://dragontts.invalid",
            model_id="elevenlabs:eleven_v3_conversational_clean_tempo",
            voice_id="voice",
            language="hi",
            params={},
            aggregate_sentences=True,
            text_filters=[],
        )
    )
    tts._max_consecutive_zero_audio_contexts = max_in_a_row
    reports: list[str] = []

    async def push_error(error_msg, **kwargs):
        reports.append(error_msg)

    monkeypatch.setattr(tts, "push_error", push_error)
    for turn in range(silent_turns):
        await tts._record_context_audio_outcome(f"ctx-{turn}", False)
    return reports


async def test_one_silent_turn_is_recognised_from_pipecats_own_message(monkeypatch):
    reports = await _pipecat_reports(monkeypatch, max_in_a_row=0, silent_turns=1)

    assert len(reports) == 1
    assert is_silent_tts_turn_report(reports[0])


async def test_a_run_of_silent_turns_that_fails_the_service_is_not_the_note(
    monkeypatch,
):
    reports = await _pipecat_reports(monkeypatch, max_in_a_row=2, silent_turns=2)

    assert len(reports) == 2
    assert is_silent_tts_turn_report(reports[0])
    assert not is_silent_tts_turn_report(reports[1])


@pytest.mark.parametrize(
    "message",
    [
        "server rejected WebSocket connection: HTTP 400",
        "Unknown error occurred: boom",
        "DragonTTS stream failed: TTS context abc completed with no audio",
        "",
    ],
)
def test_other_pipeline_errors_are_not_the_note(message):
    assert not is_silent_tts_turn_report(message)


class _Handlers:
    """Stands in for a pipecat object that takes event handlers: keeps them by name."""

    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler

        return register


def _agent_with_handlers():
    sent = []

    async def emit(event_type, payload=None):
        sent.append((event_type, payload))

    agent = SimpleNamespace(
        transport=_Handlers(),
        task=_Handlers(),
        errors=[],
        lead=None,
        is_stream_mode=False,
        _rtvi_processor=_Handlers(),
        _context_aggregator=None,
        _emit_rtvi_event=emit,
    )
    # The method only reads the attributes set above.
    Agent._register_event_handlers(cast(Any, agent))
    return agent, sent


async def test_the_note_is_neither_stored_nor_sent_to_the_client():
    agent, sent = _agent_with_handlers()
    error = SimpleNamespace(
        processor="DragonTTSService#8",
        error="TTS context 4704d033-bd55-44ff-851a-1143aea3cd73 completed with no audio",
    )

    await agent.task.handlers["on_pipeline_error"](agent.task, error)

    assert agent.errors == []
    assert sent == []


async def test_any_other_pipeline_error_is_stored_and_sent_to_the_client():
    agent, sent = _agent_with_handlers()
    error = SimpleNamespace(
        processor="ElevenLabsTTSService#1",
        error="server rejected WebSocket connection: HTTP 400",
    )

    await agent.task.handlers["on_pipeline_error"](agent.task, error)

    assert [row["message"] for row in agent.errors] == [
        "[PIPELINE] ElevenLabsTTSService#1: server rejected WebSocket connection: HTTP 400"
    ]
    assert sent == [
        (
            "pipeline-error",
            {
                "processor": "ElevenLabsTTSService#1",
                "error": "server rejected WebSocket connection: HTTP 400",
            },
        )
    ]
