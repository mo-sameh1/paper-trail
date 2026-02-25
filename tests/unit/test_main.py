from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from typing import Generator
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.main import app, get_db  # noqa: E402
from backend.interfaces import GraphDatabaseInterface  # noqa: E402


def override_get_db() -> Generator[Mock, None, None]:
    mock_db = Mock(spec=GraphDatabaseInterface)
    # Configure mock behavior if needed
    yield mock_db


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

MOCK_INFERENCE_DATA = {
    "entities": [
        {"id": "doc_1", "properties": {"type": "Document", "content": "mock content"}},
    ],
    "relationships": [
        {
            "source_id": "doc_1",
            "target_id": "test",
            "type": "MENTIONS",
            "properties": {},
        }
    ],
}


@patch("backend.main.run_inference_mock")
def test_upload_success(mock_inference: Mock) -> None:
    mock_inference.return_value = MOCK_INFERENCE_DATA

    response = client.post("/upload", json={"text": "This is a test document."})

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Text processed and relationships inserted into the graph.",
    }
    mock_inference.assert_called_once_with("This is a test document.")


def test_upload_empty_text() -> None:
    response = client.post("/upload", json={"text": ""})

    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"] == "Text cannot be empty."


@patch("backend.main.run_inference_mock")
def test_upload_db_failure(mock_inference: Mock) -> None:
    mock_inference.return_value = MOCK_INFERENCE_DATA

    # Simulate a DB insertion error by replacing the dependency override temporarily
    # Since we use app.dependency_overrides globally, we can mock the specific method
    # on the yielded mock object inside the route, but it's easier to just raise error
    # from the overridden get_db
    mock_db_failing = Mock(spec=GraphDatabaseInterface)
    mock_db_failing.insert_relationship_graph.side_effect = Exception(
        "DB Connection Refused"
    )

    app.dependency_overrides[get_db] = lambda: mock_db_failing

    try:
        response = client.post("/upload", json={"text": "Test DB failure"})

        assert response.status_code == 500
        error_msg = response.json()["detail"]
        assert "An error occurred: DB Connection Refused" in error_msg
    finally:
        # Restore normal mock
        app.dependency_overrides[get_db] = override_get_db
