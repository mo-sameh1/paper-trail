from typing import Any, Dict, List

from datasets import load_dataset  # type: ignore

from ml_pipeline.config import training_config


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
        print(f"Loading dataset: {self.dataset_name} (validation split)")
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


if __name__ == "__main__":
    processor = DocredProcessor()
    docs = processor.fetch_validation_batch()
    print(f"Fetched {len(docs)} documents.")
    if docs:
        sample = docs[0]["text"][:200] + "..."
        # Safely print avoiding Windows charmap crash
        print("Sample Text:", sample.encode("utf-8", "replace").decode("utf-8"))
