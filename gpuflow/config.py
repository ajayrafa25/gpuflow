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

    # MLflow
    MLFLOW_PORT: int = 5001
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_CONTAINER_URI: str = "http://172.17.0.1:5001"
    MLFLOW_STORE_PATH: str = "./mlruns"

    # Debug sessions (code-server)
    PUBLIC_HOST: str = "localhost"
    CODE_SERVER_PATH: str = "/usr/bin/code-server"
    DEBUG_PORT_START: int = 8090
    DEBUG_PORT_END: int = 8099


settings = Settings()
