from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """
    Configuration parameters strictly for the Language Model.
    Exposed via .env file or environment variables.
    """

    model_id: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_new_tokens: int = 50
    hf_token: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


llm_config = LLMConfig()
