"""
Evaluates the baseline untrained 4-bit model to capture preliminary
accuracy/loss metrics using the centralized LLM wrapper.
"""

import time

import torch

from llm.model import LLMManager
from llm.prompts import NERPrompts
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model() -> None:
    """
    Loads 4-bit model via LLMManager, runs test prompts, and outputs metrics.
    """
    logger.info("Loading LLM Manager in evaluation mode...")
    llm = LLMManager(is_training=False)

    logger.info("--- Baseline Evaluation ---")
    start_time = time.time()

    for i, prompt in enumerate(NERPrompts.EVALUATION_SAMPLES):
        response = llm.generate(prompt)

        logger.info(f"Prompt {i+1}: {prompt}")
        logger.info(f"Response: {response}")

    duration = time.time() - start_time
    logger.info(
        f"Total inference time for {len(NERPrompts.EVALUATION_SAMPLES)} "
        f"prompts: {duration:.2f} seconds"
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Allocated VRAM: {allocated:.2f} GB")
    else:
        logger.warning("CUDA not available. VRAM could not be measured.")


if __name__ == "__main__":
    evaluate_model()
