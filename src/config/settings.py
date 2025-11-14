"""
Configuration settings for IntelliFlow
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Alibaba Cloud API Configuration
    ali_model_name: str = "qwen-plus"
    ali_api_key: str
    ali_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "intelliflow"
    postgres_user: str = "postgres"
    postgres_password: str = "password123"
    
    # Weather API Configuration
    amap_api_key: str
    
    # RAG Configuration
    max_retrieved_docs: int = 3
    default_similarity_threshold: float = 0.7
    chunk_size: int = 300
    chunk_overlap: int = 30
    max_history_turns: int = 5
    
    # Available Models
    available_models: List[str] = ["qwen-plus", "qwen-turbo", "qwen-max"]
    
    # Text Splitter Configuration
    separators: List[str] = ["\n\n", "\n", ".", "!", "?", " ", ""]
    
    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
