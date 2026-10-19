from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and RAG configuration settings."""

    # Application
    APP_NAME: str = "enterprise-agentic-rag"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # API Keys
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Model Provider & Names
    LLM_PROVIDER: Literal["openai", "groq"] = "openai"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    GROQ_MODEL: str = "llama3-70b-8192"
    TEMPERATURE: float = 0.0

    # Vector Store Settings
    VECTOR_STORE_TYPE: Literal["chromadb", "qdrant"] = "chromadb"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    COLLECTION_NAME: str = "enterprise_knowledge_base"

    # Embeddings
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion & Splitting
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
