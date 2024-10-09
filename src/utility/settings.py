from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path
import os
env_path = os.path.abspath(Path('.env'))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_path, 
        env_file_encoding='utf-8'
    )
    WANDB_API_KEY:str

app_settings = Settings()
