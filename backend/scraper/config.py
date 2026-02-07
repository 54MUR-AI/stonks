from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    huggingface_api_key: str = Field(default="", env="HUGGINGFACE_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    summarizer_provider: Literal["openai", "anthropic", "huggingface", "ollama", "simple"] = Field(
        default="simple", env="SUMMARIZER_PROVIDER"
    )
    summarizer_model: str = Field(
        default="llama3.2", env="SUMMARIZER_MODEL"
    )
    max_summary_length: int = Field(default=500, env="MAX_SUMMARY_LENGTH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
