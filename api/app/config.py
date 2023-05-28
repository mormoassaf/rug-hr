from pydantic import BaseSettings
from functools import lru_cache
import logging


class Settings(BaseSettings):
    app_name: str
    version: str
    description: str
    whitelist: list

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


# For loading models
artifacts = {
    "device": None,
    "tokenizer": {
        "finbert": None,
        "bertweet": None
    },
    "model": {
        "finbert": None,
        "bertweet": None
    }
}

log = logging.getLogger("baas-logger")
