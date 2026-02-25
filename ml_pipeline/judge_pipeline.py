import json
import re
from typing import Any, Dict, List, Set, Tuple

import torch

from llm.model import LLMManager
from llm.prompts import NERPrompts
from ml_pipeline.process_docred import DocredProcessor


class JudgePipeline:
    """
    Evaluates the LLM's entity relation extraction capabilities against ground truth data.
    Calculates Precision, Recall, and F1-Score.
    """

    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self.processor = DocredProcessor()
        # Enforce smaller limit specifically for the evaluation loop duration
        self.processor.limit = self.limit
        self.llm = LLMManager(is_training=False)

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """
        Attempts to heavily parse the JSON block returned by the LLM.
        Avoids crashing on hallucinated surrounding text.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)  # type: ignore
            except json.JSONDecodeError:
                pass
        return {"relationships": []}

    def _get_ground_truth_triples(
        self, doc: Dict[str, Any]
    ) -> Set[Tuple[str, str, str]]:
        """
        Parses docred ground truth specifically into (subject, relation, object) tuples.
        """
        triples = set()
        labels = doc.get("labels", [])
        vertex_set = doc.get("vertexSet", [])

        if not labels or not vertex_set:
            return triples

        for label in labels:
            try:
                # docred maps entities via indices pointing to the vertexSet
                subj_idx = label.get("head")
                obj_idx = label.get("tail")
                rel_type = label.get("relation")

                if subj_idx is None or obj_idx is None or rel_type is None:
                    continue

                # get the primary string name for the entity
                subj_name = vertex_set[subj_idx][0]["name"].lower()
                obj_name = vertex_set[obj_idx][0]["name"].lower()

                triples.add((subj_name, rel_type, obj_name))
            except (IndexError, KeyError):
                continue

        return triples

    def evaluate(self) -> None:
        """
        Runs the full evaluation pipeline, tracking F1-score across documents.
        """
        docs = self.processor.fetch_validation_batch()
        if not docs:
            print("No documents fetched. Aborting evaluation.")
            return

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        print(f"Beginning evaluation on {len(docs)} documents.")

        for i, doc in enumerate(docs):
            print(f"\nEvaluating Context {i + 1}/{len(docs)}...")
            prompt = NERPrompts.RELATION_EXTRACTION_PROMPT.format(
                document_text=doc["text"]
            )

            # Request extraction up to 300 new tokens
            response_text = self.llm.generate(prompt, max_new_tokens=300)
            parsed_json = self._extract_json_from_response(response_text)

            extracted_triples = set()
            for rel in parsed_json.get("relationships", []):
                subj = str(rel.get("subject", "")).lower()
                obj = str(rel.get("object", "")).lower()
                # Simplified check: just checking if the entities match.
                # In strict evaluation, relation type would be checked securely.
                if subj and obj:
                    extracted_triples.add((subj, obj))

            ground_truth = self._get_ground_truth_triples(doc)
            # Simplify ground truth to (subject, object) to measure entity linking recall easily
            # for baseline checkout.
            gt_simplified = {(s, o) for s, r, o in ground_truth}

            true_positives = len(extracted_triples.intersection(gt_simplified))
            false_positives = len(extracted_triples - gt_simplified)
            false_negatives = len(gt_simplified - extracted_triples)

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            print(
                f"--> Extracted: {len(extracted_triples)} | Got Right: {true_positives}"
            )
            print(f"--> Ground Truths Total: {len(ground_truth)}")

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

        print("\n" + "=" * 40)
        print("INITIAL PRELIMINARY RESULTS (CHECKPOINT 1)")
        print("=" * 40)
        print(f"Documents Evaluated: {len(docs)}")
        print(f"Total True Positives : {total_true_positives}")
        print(f"Total False Positives: {total_false_positives}")
        print(f"Total False Negatives: {total_false_negatives}")
        print("-" * 40)
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-Score   : {f1_score:.4f}")

        if torch.cuda.is_available():
            print(f"\nVRAM Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    judge = JudgePipeline(limit=2)
    judge.evaluate()
