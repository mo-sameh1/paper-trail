import json
from typing import Dict, List

from datasets import load_dataset  # type: ignore

from llm.prompts import NERPrompts
from ml_pipeline.config import training_config


def format_legal_ner_dataset(output_path: str) -> None:
    """
    Downloads `legal-ner` dataset and formats it for instruction tuning.
    """
    dataset_name = training_config.dataset_name
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    formatted_data: List[Dict[str, str]] = []
    max_samples = training_config.max_samples

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        instruction = NERPrompts.SYSTEM_INSTRUCTION

        # legal-ner usually provides tokens and ner_tags
        if "tokens" in example and "ner_tags" in example:
            text = " ".join(example["tokens"])
            output = f"NER Tags: {example.get('ner_tags', [])}"
        elif "text" in example:
            text = example["text"]
            output = f"Entities: {example.get('entities', 'None')}"
        else:
            text = str(example)
            output = "Extracted entities placeholder."

        formatted_data.append(
            {
                "instruction": instruction,
                "input": text,
                "output": output,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Successfully formatted {len(formatted_data)} samples to {output_path}")


if __name__ == "__main__":
    format_legal_ner_dataset("formatted_instructions.json")
