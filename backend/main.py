from typing import Generator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from backend.config import db_config
from backend.database import Neo4jGraphDatabase
from backend.interfaces import GraphDatabaseInterface
from llm.inference import InferenceService

app = FastAPI(title="FOIA Dump Investigative Grapher API")
inference_service = InferenceService()


# Dependency Injection for Database
def get_db() -> Generator[GraphDatabaseInterface, None, None]:
    db = Neo4jGraphDatabase(db_config.uri, db_config.user, db_config.password)
    try:
        yield db
    finally:
        db.close()


class UploadRequest(BaseModel):
    text: str


class UploadResponse(BaseModel):
    status: str
    message: str


@app.post("/upload", response_model=UploadResponse)
def upload_text(request: UploadRequest, db: GraphDatabaseInterface = Depends(get_db)):
    """
    Accepts text, runs inference to extract entities/relations, and stores in Neo4j.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # 1. Run inference via decoupled service
        graph_data = inference_service.extract_graph(request.text)

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
