import os
from typing import Any, Dict, Generator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from backend.database import Neo4jGraphDatabase
from backend.interfaces import GraphDatabaseInterface

app = FastAPI(title="FOIA Dump Investigative Grapher API")

# Neo4j configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# Dependency Injection for Database
def get_db() -> Generator[GraphDatabaseInterface, None, None]:
    db = Neo4jGraphDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        yield db
    finally:
        db.close()


class UploadRequest(BaseModel):
    text: str


class UploadResponse(BaseModel):
    status: str
    message: str


def run_inference_mock(text: str) -> Dict[str, Any]:
    """
    Mock inference function representing the ML pipeline output.
    Returns structured JSON for Graph insertion.
    """
    # In a real scenario, this would format the prompt, call the Llama-3 model,
    # and parse the generated JSON output.
    return {
        "entities": [
            {"id": "doc_1", "properties": {"type": "Document", "content": text}},
            {"id": "org_cia", "properties": {"type": "Organization", "name": "CIA"}},
        ],
        "relationships": [
            {
                "source_id": "doc_1",
                "target_id": "org_cia",
                "type": "MENTIONS",
                "properties": {"confidence": 0.95},
            }
        ],
    }


@app.post("/upload", response_model=UploadResponse)
def upload_text(request: UploadRequest, db: GraphDatabaseInterface = Depends(get_db)):
    """
    Accepts text, runs inference to extract entities/relations, and stores in Neo4j.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # 1. Run inference (mocked for now)
        graph_data = run_inference_mock(request.text)

        # 2. Insert into Neo4j
        db.insert_relationship_graph(
            entities=graph_data["entities"],
            relationships=graph_data["relationships"],
        )

        return UploadResponse(
            status="success",
            message="Text processed and relationships inserted into the graph.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
