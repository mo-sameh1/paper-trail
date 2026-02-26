"""
Demonstration script for Checkpoint 1.
Pulls relation data from the LLM based on source documents, inserts it into
the Graph Database, and executes a test query to prove functionality.
"""

from backend.config import db_config
from backend.database import Neo4jGraphDatabase
from llm.inference import InferenceService
from ml_pipeline.process_docred import DocredProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphExtractionDemo:
    def __init__(self, limit: int = 5) -> None:
        self.processor = DocredProcessor()
        self.processor.limit = limit
        # The inference service manages LLM requests implicitly
        self.inference = InferenceService()
        self.db = Neo4jGraphDatabase(db_config.uri, db_config.user, db_config.password)

    def run_demo(self) -> None:
        logger.info("Starting Graph DB Extraction & Storage Demo.")
        logger.info("Clearning graph database for fresh checkout environment...")

        # Neo4jGraphDatabase abstracts query execution usually, but we can do
        # a raw clear here:
        with self.db._driver.session() as session:  # type: ignore
            session.run("MATCH (n) DETACH DELETE n")

        docs = self.processor.fetch_validation_batch()

        # 1. Process and Insert
        for i, doc in enumerate(docs):
            logger.info(f"Processing Document {i + 1}/{len(docs)}...")

            # Currently this hits the inference mock/wrapper.
            # This demonstrates the modular integration point.
            graph_data = self.inference.extract_graph(doc["text"])

            # Since extract_graph returns a structured dict
            self.db.insert_relationship_graph(
                entities=graph_data["entities"],
                relationships=graph_data["relationships"],
            )
            logger.info("Successfully inserted nodes and edges.")

        # 2. Test Cypher Query
        logger.info("--- Querying the Embedded Graph Data ---")
        query = (
            "MATCH (s)-[r]->(o) "
            "RETURN s.name AS subject, type(r) AS relation, o.name AS object "
            "LIMIT 10"
        )
        with self.db._driver.session() as session:  # type: ignore
            result = session.run(query)
            for record in result:
                subject = record.get("subject", "Unknown_Subj")
                relation = record.get("relation", "RELATION")
                obj = record.get("object", "Unknown_Obj")
                logger.info(f"Found Edge: ({subject}) -[{relation}]-> ({obj})")

        logger.info("Demo complete.")
        self.db.close()


if __name__ == "__main__":
    demo = GraphExtractionDemo(limit=2)
    demo.run_demo()
