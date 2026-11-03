from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_oai_endpoint: str = ""
    azure_oai_key: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_oai_embedding_deployment: str = "text-embedding-3-small"
    azure_oai_embedding_deployment_large: str = ""

    embedding_store_directory: str = "./data/embeddings"
    chroma_persist_directory: str = "./data/chroma_db"
    ground_truth_csv: str = "./Ground truth Onet to Esco/ONET_(Occupations)_0_updated.csv"
    cors_origins: str = "http://localhost:5173"
    api_port: int = 8001

    # TypeSafe System One API (https://docs.typesafe.ai/api)
    jev_base_url: str = "https://api.typesafe.ai"
    jev_api_key: str = ""
    jev_concurrency: int = 16

    # Raw taxonomy sources for gated JEV enrichment (jev_context_v2)
    onet_sql_dir: str = ""
    esco_csv_dir: str = ""

    @property
    def embedding_deployment(self) -> str:
        return (
            self.azure_oai_embedding_deployment_large.strip()
            or self.azure_oai_embedding_deployment
        )

    @property
    def embedding_model_name(self) -> str:
        dep = self.embedding_deployment.lower()
        if "large" in dep:
            return "text-embedding-3-large"
        return "text-embedding-3-small"

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def embedding_dir(self) -> Path:
        p = Path(self.embedding_store_directory)
        return p if p.is_absolute() else ROOT / p

    @property
    def chroma_dir(self) -> Path:
        p = Path(self.chroma_persist_directory)
        return p if p.is_absolute() else ROOT / p

    @property
    def ground_truth_path(self) -> Path:
        p = Path(self.ground_truth_csv)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
