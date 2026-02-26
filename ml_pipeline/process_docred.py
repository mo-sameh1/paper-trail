from typing import Any, Dict, List, Set, Tuple

from datasets import load_dataset  # type: ignore

from ml_pipeline.config import training_config
from utils.logger import get_logger

logger = get_logger(__name__)


class DocredProcessor:
    """
    Handles fetching and formatting the docred dataset for relation extraction
    evaluation.
    """

    def __init__(self) -> None:
        self.dataset_name = training_config.dataset_name
        self.limit = training_config.docred_sample_limit

    def fetch_validation_batch(self) -> List[Dict[str, Any]]:
        """
        Fetches a subset of the docred validation split.
        Transforms the 'sents' (list of lists of words) into a single document string.
        Returns a list of dicts with 'text' and ground-truth 'labels'.
        """
        logger.info(f"Loading dataset: {self.dataset_name} (validation split)")
        dataset = load_dataset(self.dataset_name, split="validation")

        batch: List[Dict[str, Any]] = []
        for i, example in enumerate(dataset):
            if i >= self.limit:
                break

            # Flatten the sentences into a single document string
            # 'sents' is a list of sentences, where each sentence is a list of words
            sentences = [" ".join(sent) for sent in example["sents"]]  # type: ignore
            doc_text = " ".join(sentences)

            batch.append(
                {
                    "title": example.get("title", f"Doc_{i}"),
                    "text": doc_text,
                    "vertexSet": example.get("vertexSet", []),
                    "labels": example.get("labels", []),
                }
            )

        return batch

    @staticmethod
    def get_ground_truth_triples(
        doc: Dict[str, Any],
    ) -> Set[Tuple[str, str, str]]:
        """
        Parses DocRED ground truth into (subject, relation, object) triples.
        Uses vertexSet indices referenced by labels to resolve entity names.
        """
        triples: Set[Tuple[str, str, str]] = set()
        labels = doc.get("labels", [])
        vertex_set = doc.get("vertexSet", [])

        if not labels or not vertex_set:
            return triples

        for label in labels:
            try:
                # Re-DocRED uses short keys: h (head), t (tail), r (relation)
                subj_idx = label.get("h", label.get("head"))
                obj_idx = label.get("t", label.get("tail"))
                rel_type = label.get("r", label.get("relation", ""))

                if subj_idx is None or obj_idx is None or not rel_type:
                    continue

                subj_name = vertex_set[subj_idx][0]["name"].lower()
                obj_name = vertex_set[obj_idx][0]["name"].lower()

                triples.add((subj_name, rel_type, obj_name))
            except (IndexError, KeyError):
                continue

        return triples


if __name__ == "__main__":
    processor = DocredProcessor()
    docs = processor.fetch_validation_batch()
    logger.info(f"Fetched {len(docs)} documents.")
    if docs:
        sample = docs[0]["text"][:200] + "..."
        # Safely print avoiding Windows charmap crash
        clean_sample = sample.encode("utf-8", "replace").decode("utf-8")
        logger.debug(f"Sample Text: {clean_sample}")
