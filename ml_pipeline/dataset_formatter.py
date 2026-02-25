"""
Loads a subset of the Hugging Face `legal-ner` dataset and formats it into
instruction-response pairs suitable for LLM fine-tuning.
"""

import json
from typing import Dict, List

from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv

load_dotenv()


def format_legal_ner_dataset(output_path: str, max_samples: int = 1000) -> None:
    """
    Downloads `legal-ner` dataset and formats it for instruction tuning.
    """
    print("Loading dataset: daishen/legal-ner")
    try:
        dataset = load_dataset("daishen/legal-ner", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    formatted_data: List[Dict[str, str]] = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        instruction = (
            "Extract the named entities (PERSON, ORGANIZATION, LOCATION) "
            "from the following legal text."
        )

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
