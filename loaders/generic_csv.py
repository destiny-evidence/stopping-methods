import logging
from pathlib import Path

import pandas as pd
from shared.collection import AbstractCollection
from shared.config import settings
from shared.dataset import Dataset

logger = logging.getLogger(__name__)


class GenericCollection(AbstractCollection):
    BASE: str = 'generic-csv'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        files = list(self.raw_folder.glob('*.csv'))
        logger.info(f'Searching for CSVs in {self.raw_folder}')
        logger.info(f'Found files: {files}')
        for file in files:
            yield read_file(file, f'generic-csv-{file.stem}')


def read_file(file_path: Path, key: str) -> Dataset:
    df = pd.read_csv(file_path).fillna('')
    df = df[((df['label_abs'] == 0) | (df['label_abs'] == 0)) & df['abstract'].str.len() > 0]

    return Dataset(
        key=key,
        labels=[rec['label_abs'] for _, rec in df.iterrows()],
        texts=[(rec['title'] or '') + ' ' + (rec['abstract'] or '') for _, rec in df.iterrows()]
    )


def read_csv_dataset(key: str) -> Dataset:
    base = GenericCollection.BASE
    base_dir = settings.raw_data_path / base
    base_name = key[len(base) + 1:]
    file_path = base_dir / f'{base_name}.csv'
    if not file_path.exists():
        raise AssertionError(f'Files for {key} not valid: {file_path}')

    return read_file(file_path, key)
