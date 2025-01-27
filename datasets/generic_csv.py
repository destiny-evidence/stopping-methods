import logging

import pandas as pd
from shared.collection import AbstractCollection
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
            df = pd.read_csv(file).fillna('')
            yield Dataset(
                key=f'generic-csv-{file.stem}',
                labels=[rec['label_abs'] for _, rec in df.iterrows()],
                texts=[(rec['title'] or '') + ' ' + (rec['abstract'] or '') for _, rec in df.iterrows()]
            )
