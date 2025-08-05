from functools import lru_cache

from pydantic_settings import BaseSettings

@lru_cache
class AppSettings(BaseSettings):
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: str | None = None
    POSTGRES_URI: str | None = None
    class Config:
        env_file: str = ".env"


settings = AppSettings()