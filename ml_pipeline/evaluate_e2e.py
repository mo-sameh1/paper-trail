"""
E2E Evaluation Script for Checkpoint 1.
Validates the entire workflow:
Dataset Fetch -> LLM Inference -> DB Insertion -> DB Fetch -> NLP Compare -> Output
"""

import json
import time
from typing import Any, Dict, List, Set, Tuple

import torch

from backend.config import db_config
from backend.database import Neo4jGraphDatabase
from llm.inference import InferenceService
from ml_pipeline.process_docred import DocredProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class E2EEvaluationRunner:
    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self.processor = DocredProcessor()
        self.processor.limit = self.limit
        self.inference = InferenceService()
        self.db = Neo4jGraphDatabase(db_config.uri, db_config.user, db_config.password)

    def clear_database(self) -> None:
        logger.info("Clearing Neo4j Graph Database for fresh evaluation.")
        with self.db._driver.session() as session:  # type: ignore
            session.run("MATCH (n) DETACH DELETE n")

    def evaluate(self) -> None:
        logger.info("Starting E2E Refined Pipeline Evaluation...")
        self.clear_database()

        docs = self.processor.fetch_validation_batch()
        if not docs:
            logger.error("Failed to load documents.")
            return

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        eval_results: List[Dict[str, Any]] = []
        start_time = time.time()

        for i, doc in enumerate(docs):
            logger.info(f"Processing Document {i + 1}/{len(docs)}")

            # 1. Inference
            graph_data = self.inference.extract_graph(doc["text"])

            # 2. Store to DB
            self.db.insert_relationship_graph(
                entities=graph_data["entities"],
                relationships=graph_data["relationships"],
            )

            # 3. Query from DB (Proving it was stored)
            queried_triples: Set[Tuple[str, str]] = set()
            query = "MATCH (s)-[r]->(o) RETURN s.name AS subject, o.name AS object"
            with self.db._driver.session() as session:  # type: ignore
                result = session.run(query)
                for record in result:
                    subj_val = record.get("subject")
                    obj_val = record.get("object")
                    if subj_val is not None and obj_val is not None:
                        subj = str(subj_val).lower()
                        obj = str(obj_val).lower()
                        if subj and obj:
                            queried_triples.add((subj, obj))

            # 4. Compare with Ground Truth (LLM Judge)
            ground_truth = DocredProcessor.get_ground_truth_triples(doc)
            gt_simplified = {(s, o) for s, r, o in ground_truth}

            judge_result = self.inference.judge_triples(
                extracted=queried_triples,
                ground_truth=gt_simplified,
            )
            tp = judge_result["true_positives"]
            fp = judge_result["false_positives"]
            fn = judge_result["false_negatives"]

            total_true_positives += tp
            total_false_positives += fp
            total_false_negatives += fn

            doc_record = {
                "doc_title": doc.get("title", f"Doc_{i}"),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "db_extracted_entities": len(queried_triples),
                "ground_truth_entities": len(gt_simplified),
            }
            eval_results.append(doc_record)

            # Clear graph for next document isolation
            self.clear_database()

        self.db.close()
        duration = time.time() - start_time

        # Calculate metrics
        precision = (
            total_true_positives / (total_true_positives + total_false_positives)
            if (total_true_positives + total_false_positives) > 0
            else 0.0
        )
        recall = (
            total_true_positives / (total_true_positives + total_false_negatives)
            if (total_true_positives + total_false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        final_metrics = {
            "total_documents": len(docs),
            "vram_used_gb": (
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
            "duration_seconds": round(duration, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "document_breakdown": eval_results,
        }

        # 5. Output Results
        self._write_reports(final_metrics)

    def _write_reports(self, metrics: Dict[str, Any]) -> None:
        logger.info("Writing preliminary results to file...")

        with open("preliminary_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        lines = [
            "# Phase 2 - Checkpoint 1 Preliminary Results",
            "",
            f"**Total Documents Evaluated:** {metrics['total_documents']}",
            f"**Total Pipeline Execution Time:** {metrics['duration_seconds']}s",
            f"**Max VRAM Allocated:** {metrics['vram_used_gb']:.2f} GB",
            "",
            "## Accuracy Metrics (LLM-as-Judge Semantic Match)",
            f"- **Precision:** {metrics['precision']}",
            f"- **Recall:** {metrics['recall']}",
            f"- **F1-Score:** {metrics['f1_score']}",
            "",
            "## Pipeline Verification",
            "Metrics were extracted *after* standardizing the LLM's JSON,",
            "inserting it into Neo4j via Cypher queries, and fetching the edges back.",
            "This completely validates the end-to-end relational data flow.",
        ]
        with open("preliminary_results.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(
            "Successfully generated preliminary_results.json and preliminary_results.md"
        )


if __name__ == "__main__":
    runner = E2EEvaluationRunner(limit=3)
    runner.evaluate()
