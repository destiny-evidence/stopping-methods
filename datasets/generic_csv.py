import pandas as pd
from shared.collection import AbstractCollection
from shared.dataset import Dataset


class GenericCollection(AbstractCollection):
    BASE = 'generic-csv'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        for file in self.raw_folder.glob('*.csv'):
            df = pd.read_csv(file)
            yield Dataset(
                key=f'generic-{file.stem}',
                labels=[rec['label_abs'] for _, rec in df.iterrows()],
                texts=[(rec['title'] or '') + ' ' + (rec['abstract'] or '') for _, rec in df.iterrows()]
            )
