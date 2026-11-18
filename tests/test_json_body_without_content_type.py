import pytest
from fastapi.testclient import TestClient

from app.main import app

# Routes that take a JSON body and need no login. A page on another site can
# post to these from a visitor's browser, and a body with no Content-Type needs
# no preflight, so these do not parse one.
PUBLIC_DEMO_ROUTES = [
    "/agent/voice/breeze-buddy/demo/connect",
    "/agent/voice/breeze-buddy/chat/demo/session",
]


def test_a_json_body_with_no_content_type_is_still_parsed():
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/agent/voice/breeze-buddy/login",
        content=b'{"username": "a"}',
    )

    assert "content-type" not in response.request.headers
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "password"]


@pytest.mark.parametrize("path", PUBLIC_DEMO_ROUTES)
def test_a_public_demo_route_does_not_parse_a_body_with_no_content_type(path):
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(path, content=b'{"unknown": 1}')

    assert "content-type" not in response.request.headers
    assert response.status_code == 422
    # The body was never read as JSON: the error is about the body as a whole,
    # not about a field inside it.
    assert response.json()["detail"][0]["loc"] == ["body"]


@pytest.mark.parametrize("path", PUBLIC_DEMO_ROUTES)
def test_a_public_demo_route_parses_a_body_sent_as_json(path):
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(path, json={"unknown": 1})

    assert response.status_code == 422
    # Parsed, then refused for a missing field.
    error = response.json()["detail"][0]
    assert error["type"] == "missing"
    assert error["loc"][0] == "body" and len(error["loc"]) == 2
