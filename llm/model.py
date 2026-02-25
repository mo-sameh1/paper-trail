import torch
from peft import get_peft_model  # type: ignore
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer  # type: ignore
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from llm.config import llm_config


class LLMManager:
    """
    Central abstraction for loading, training, and running the target LLM.
    Handles huggingface instantiations efficiently to remove arbitrary code
    from the API handlers.
    """

    def __init__(self, is_training: bool = False) -> None:
        self.is_training = is_training

        # Consistent quantization settings specifically for 8GB VRAM
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_config.model_id, token=llm_config.hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_config.model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            token=llm_config.hf_token,
        )

        if self.is_training:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Simple text generation interface for the backend.
        """
        if max_new_tokens is None:
            max_new_tokens = llm_config.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return str(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

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
