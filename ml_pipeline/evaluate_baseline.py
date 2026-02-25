"""
Evaluates the baseline untrained 4-bit model to capture preliminary
accuracy/loss metrics.
"""

import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def evaluate_model() -> None:
    """
    Loads 4-bit model, runs test prompts, and outputs metrics.
    """
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    prompts = [
        "Extract named entities from this text: The CIA employed John Doe in 2022.",  # noqa: E501
        "Generate a JSON graph of entities and relations from: OpenAI was founded by Sam Altman.",  # noqa: E501
    ]

    print("\n--- Baseline Evaluation ---")
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {response}")

    duration = time.time() - start_time
    print(f"\nTotal inference time for {len(prompts)} prompts: {duration:.2f} seconds")

    if torch.cuda.is_available():
        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CUDA not available. VRAM could not be measured.")


if __name__ == "__main__":
    evaluate_model()
