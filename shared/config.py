import logging
import os
from pathlib import Path

import pandas as pd
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
    def ranking_data_path(self) -> Path:
        return self.DATA_PATH / 'rankings'

    @property
    def result_data_path(self) -> Path:
        return self.DATA_PATH / 'results'


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)

pd.options.display.max_columns = None
pd.options.display.max_rows = None

conf_file = os.environ.get('STOP_CONF', 'config/default.env')
settings = Settings(_env_file=conf_file, _env_file_encoding='utf-8')

__all__ = ['settings', 'conf_file']
