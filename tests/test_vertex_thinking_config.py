"""build_vertex_llm hands pipecat only the thinking field the template set.

pipecat sends the fields it was handed (``model_dump(exclude_unset=True)``),
so an empty ``thinking_level`` still reaches the request as a key. Chat's
per-turn thinking override looks for that key, and with it present adds a
level next to the budget; Gemini rejects a request that carries both.
"""

from __future__ import annotations

import pytest
from pipecat.services.google.llm import GoogleLLMService

from app.ai.voice.llm import vertex
from app.ai.voice.llm.vertex import VertexConfig, build_vertex_llm


@pytest.fixture
def settings_sent(monkeypatch) -> dict:
    """What build_vertex_llm passes to the Vertex service; no client is built."""
    sent: dict = {}

    def _service(**kwargs):
        sent.update(kwargs)
        return object()

    monkeypatch.setattr(vertex, "GoogleVertexLLMService", _service)
    return sent


def _config(**thinking) -> VertexConfig:
    return VertexConfig(
        credentials_json="{}",
        project_id="project",
        location="global",
        model="gemini-2.5-flash",
        temperature=0.6,
        max_tokens=500,
        **thinking,
    )


def _thinking_as_pipecat_sends_it(sent: dict) -> dict:
    return sent["settings"].thinking.model_dump(exclude_unset=True)


def test_a_budget_is_sent_without_a_level(settings_sent):
    build_vertex_llm(_config(thinking_budget=1024))
    assert _thinking_as_pipecat_sends_it(settings_sent) == {"thinking_budget": 1024}


def test_a_level_is_sent_without_a_budget(settings_sent):
    build_vertex_llm(_config(thinking_level="medium"))
    assert _thinking_as_pipecat_sends_it(settings_sent) == {"thinking_level": "medium"}


def test_no_thinking_setting_sends_no_thinking_config(settings_sent):
    build_vertex_llm(_config())
    assert not isinstance(
        settings_sent["settings"].thinking, GoogleLLMService.ThinkingConfig
    )
