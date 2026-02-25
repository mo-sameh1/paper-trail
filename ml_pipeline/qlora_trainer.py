"""
QLoRA fine-tuning script utilizing centralized LLM components.
"""

from datasets import load_dataset  # type: ignore
from transformers import TrainingArguments  # type: ignore
from trl import SFTTrainer  # type: ignore

from llm.model import LLMManager
from llm.prompts import NERPrompts
from ml_pipeline.config import training_config


def train_qlora(data_path: str) -> None:
    """
    Configures and starts QLoRA training with strict memory optimizations.
    """
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")

    llm = LLMManager(is_training=True)
    llm.attach_lora_adapter(
        r=training_config.lora_r,
        alpha=training_config.lora_alpha,
        dropout=training_config.lora_dropout,
    )

    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=training_config.learning_rate,
        max_steps=training_config.max_steps,
        fp16=True,
        save_steps=training_config.save_steps,
        remove_unused_columns=False,
    )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["instruction"])):
            text = NERPrompts.format_training_prompt(
                example["instruction"][i], example["input"][i], example["output"][i]
            )
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=llm.model,
        tokenizer=llm.tokenizer,  # type: ignore
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("Starting QLoRA training...")
    trainer.train()

    output_dir = training_config.output_dir
    trainer.model.save_pretrained(f"{output_dir}/final_adapter")  # type: ignore
    print("Training complete. Adapter saved.")


if __name__ == "__main__":
    train_qlora("formatted_instructions.json")
