"""
QLoRA fine-tuning script optimized for 8GB VRAM (NVIDIA RTX 4070).
"""

import torch
from datasets import load_dataset  # type: ignore
from peft import (  # type: ignore
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer  # type: ignore
from dotenv import load_dotenv

load_dotenv()

# Base model suitable for 8GB VRAM (4-bit quantized by default)
MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def train_qlora(data_path: str, output_dir: str = "./qlora-out") -> None:
    """
    Configures and starts QLoRA training with strict memory optimizations.
    """
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")

    # 4-bit quantization config (optimizing for 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    # LoRA config targeting q_proj and v_proj
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)  # type: ignore
    model.print_trainable_parameters()

    # Training args strictly configured to avoid OOM
    # paged_adamw_8bit offloads optimizer states to CPU when needed
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        max_steps=50,
        fp16=True,
        save_steps=25,
        remove_unused_columns=False,
    )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["instruction"])):
            text = (
                f"Instruction: {example['instruction'][i]}\n"
                f"Input: {example['input'][i]}\n"
                f"Output: {example['output'][i]}"
            )
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
    )

    print("Starting QLoRA training...")
    trainer.train()
    trainer.model.save_pretrained(f"{output_dir}/final_adapter")  # type: ignore
    print("Training complete. Adapter saved.")


if __name__ == "__main__":
    train_qlora("formatted_instructions.json")
