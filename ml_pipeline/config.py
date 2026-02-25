from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """
    Configuration parameters strictly for the ML Training Pipeline.
    """

    dataset_name: str = "daishen/legal-ner"
    max_samples: int = 1000

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 50
    save_steps: int = 25
    output_dir: str = "./qlora-out"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


training_config = TrainingConfig()
