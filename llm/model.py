import torch
from peft import get_peft_model  # type: ignore
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer  # type: ignore
from transformers import AutoModelForCausalLM

from llm.config import llm_config
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """
    Central abstraction for loading, training, and running the target LLM.
    Handles huggingface instantiations efficiently to remove arbitrary code
    from the API handlers.
    """

    def __init__(self, is_training: bool = False) -> None:
        self.is_training = is_training

        common_kwargs = {
            "pretrained_model_name_or_path": llm_config.model_id,
            "token": llm_config.hf_token,
            "cache_dir": llm_config.cache_dir,
        }

        # Try loading from local cache first; fall back to downloading.
        try:
            logger.info(f"Loading tokenizer for {llm_config.model_id} from cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                **common_kwargs, local_files_only=True
            )
        except OSError:
            logger.info("Tokenizer not in cache, downloading...")
            self.tokenizer = AutoTokenizer.from_pretrained(**common_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            logger.info(
                f"Loading model weights for {llm_config.model_id} from cache..."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs, device_map="auto", local_files_only=True  # type: ignore
            )
        except OSError:
            logger.info("Model not in cache, downloading...")
            self.model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs, device_map="auto"  # type: ignore
            )
        logger.info("Model loaded successfully.")

        if self.is_training:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()

    def format_chat_prompt(self, system_message: str, user_message: str) -> str:
        """
        Formats a system + user message pair using the tokenizer's built-in
        chat template.  Works automatically for any instruct model (Llama,
        Mistral, Zephyr, etc.) without hardcoding special tokens.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Simple text generation interface for the backend.
        Accepts a pre-formatted prompt string (already chat-templated).
        """
        if max_new_tokens is None:
            max_new_tokens = llm_config.max_new_tokens

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Only decode the newly generated tokens (strip the prompt)
        new_tokens = outputs[0][input_length:]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def attach_lora_adapter(
        self, r: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        """
        Attaches the PEFT LoRA adapter for training in the ML pipeline.
        """
        if not self.is_training:
            raise RuntimeError("Model must be loaded in training mode to attach LoRA.")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)  # type: ignore
        self.model.print_trainable_parameters()  # type: ignore
