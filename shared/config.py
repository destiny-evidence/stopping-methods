import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: Path = Path('./data')

    @property
    def raw_data_path(self) -> Path:
        return self.DATA_PATH / 'raw'

    @property
    def processed_data_path(self) -> Path:
        return self.DATA_PATH / 'processed'

    @property
    def result_data_path(self) -> Path:
        return self.DATA_PATH / 'results'


conf_file = os.environ.get('STOP_CONF', 'config/default.env')
settings = Settings(_env_file=conf_file, _env_file_encoding='utf-8')

__all__ = ['settings', 'conf_file']
