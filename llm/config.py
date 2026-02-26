import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """
    Configuration parameters strictly for the Language Model.
    Exposed via .env file or environment variables.
    """

    model_id: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_new_tokens: int = 50
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    cache_dir: str = Field(default="./hf_cache", alias="HF_CACHE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )


llm_config = LLMConfig()

# Configure Hugging Face explicitly from the .env parameters
os.environ["HF_HOME"] = os.path.abspath(llm_config.cache_dir)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if llm_config.hf_token:
    os.environ["HF_TOKEN"] = llm_config.hf_token
