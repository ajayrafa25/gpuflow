from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    API_KEY: str = "dev-change-me"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    DB_PATH: str = "./gpuflow.db"
    LOG_DIR: str = "./logs"

    DEFAULT_DOCKER_IMAGE: str = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
    SCHEDULER_POLL_INTERVAL: float = 2.0
    MAX_GPUS_PER_JOB: int = 16

    GPUFLOW_SERVER_URL: str = "http://localhost:8000"


settings = Settings()
