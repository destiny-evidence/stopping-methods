import re
import json
import logging
from pathlib import Path
from typing import Generator

import pandas as pd

from shared.collection import AbstractCollection
from shared.config import settings
from shared.dataset import Dataset
from shared.pubmed import fetch
from shared.util import batched

logger = logging.getLogger('CLEF-TAR')
logging.getLogger('httpcore').setLevel('WARNING')


class CLEFCollection(AbstractCollection):
    BASE: str = 'clef'

    @property
    def folder_datasets(self):
        return self.raw_folder / 'datasets'

    def generate_datasets(self) -> Generator[Dataset, None, None]:
        files = list(self.folder_datasets.glob('*.jsonl'))
        logger.info(f'Searching for jsonl dumps in {self.folder_datasets}')
        logger.info(f'Found {len(files)} files')
        for file in files:
            yield read_file(file, key=f'{self.BASE}-{file.stem}')

    def prepare_datasets(self) -> None:
        pass

    def fetch_collection(self):
        logger.info('Fetching CLEF TAR collection')

        # Ensure target folder exists
        self.folder_datasets.mkdir(parents=True, exist_ok=True)

        for challenge in self.raw_folder.glob('*.csv'):
            df = pd.read_csv(challenge)
            for rev, items in df.groupby('topic'):
                logger.info(f'Retrieving records for {rev}')
                file_name = self.folder_datasets / f'{rev}.jsonl'
                logger.debug(f'  > {rev} @ {file_name}')

                if file_name.exists():
                    logger.info(f'Skipping {rev} which already exists in folder {self.folder_datasets}')
                    continue

                try:
                    with open(file_name, 'w') as f_out:
                        for batch in batched(items.to_dict(orient='records'), batch_size=50):
                            batch_ids = [{'pubmed_id': rec['study']} for rec in batch]
                            lookup = {rec['study']: rec for rec in batch}
                            for record in fetch(ids=batch_ids):
                                f_out.write(json.dumps(record | lookup[int(record['pubmed_id'])]) + '\n')
                except Exception as e:
                    logger.error(f'Error fetching {rev}: {e}')
                    logger.exception(e)
                    file_name.unlink(missing_ok=True)


def read_file(file_path: Path, key: str) -> Dataset:
    with open(file_path, 'r') as f:
        records = [json.loads(line) for line in f]
        records = [
            rec for rec in records
            if (rec.get('label_abs') == 0 or rec.get('label_abs') == 1)
               and rec.get('abstract') is not None
               and len(rec.get('abstract')) > 0
        ]
        return Dataset(
            key=key,
            labels=[int(rec['label_abs']) for rec in records],
            texts=[(rec['title'] or '') + ' ' + (rec['abstract'] or '') for rec in records]
        )


def read_clef_dataset(key: str) -> Dataset:
    base = CLEFCollection.BASE
    base_dir = settings.raw_data_path / base / 'datasets'
    base_name = key[len(base) + 1:]
    file_path = base_dir / f'{base_name}.jsonl'
    if not file_path.exists():
        raise AssertionError(f'Files for {key} not valid: {file_path}')

    return read_file(file_path, key)


def consolidate_files():
    for year in [2017, 2018, 2019]:
        for topic in ['prognosis', 'int', 'intervention', 'qualitative', 'dta', 'task2', '2017']:
            buffer = []
            for part in ['train', 'test']:
                for label in ['abs', 'content']:
                    fn = Path(f'data/raw/clef/qrels/full.{part}.{topic}.{label}.{year}.qrels')
                    print(fn)
                    if fn.exists():
                        df = pd.read_csv(str(fn), sep=r'\s+')
                        df.columns = ['topic', 'ignore', 'study', ('label_abs' if label == 'abs' else 'label_ft')]
                        df = df.drop(columns=['ignore'])
                        df['part'] = 1 if part == 'train' else 0
                        buffer.append(df)
            if len(buffer) > 0:
                (
                    pd
                    .concat(buffer)
                    .groupby(['topic', 'study'])
                    .max()
                    .reset_index()
                    .astype({'label_ft': 'Int8', 'label_abs': 'Int8', 'part': 'Int8'})
                    .to_csv(f'data/raw/clef/{year}-{topic}.csv', index=False)
                )


if __name__ == '__main__':
    print("In module products __package__, __name__ ==", __package__, __name__)
    __package__ = 'loaders.clef'

    if False:
        consolidate_files()

    CLEFCollection().fetch_collection()
