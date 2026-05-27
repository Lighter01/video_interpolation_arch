from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dataset_root: Path = Field(default=Path("datasets"), alias="DATASET_ROOT")
    model_repos_root: Path = Field(default=Path("model_repos"), alias="MODEL_REPOS_ROOT")
    model_weights_root: Path = Field(default=Path("model_weights"), alias="MODEL_WEIGHTS_ROOT")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI",
    )

    def resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return project_root() / path

    @property
    def dataset_root_abs(self) -> Path:
        return self.resolve_path(self.dataset_root)

    @property
    def model_repos_root_abs(self) -> Path:
        return self.resolve_path(self.model_repos_root)

    @property
    def model_weights_root_abs(self) -> Path:
        return self.resolve_path(self.model_weights_root)

    def model_repo_path(self, repo_name: str) -> Path:
        return self.model_repos_root_abs / repo_name

    def model_weight_path(self, *parts: str) -> Path:
        return self.model_weights_root_abs.joinpath(*parts)


def load_settings() -> Settings:
    return Settings()
