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

logger = logging.getLogger('SIGIR2017')
logging.getLogger('httpcore').setLevel('WARNING')

ID_MATCH = re.compile(r'(CD.*?)(:?\.|/)')


class SIGIRCollection(AbstractCollection):
    BASE: str = 'sigir-2017'

    @property
    def citations(self):
        return pd.DataFrame.from_records(json.load(open(self.raw_folder / 'citations.json')))

    @property
    def reviews(self):
        return pd.DataFrame.from_records(json.load(open(self.raw_folder / 'reviews.json'))).set_index('id')

    @property
    def folder_datasets(self):
        return self.raw_folder / 'datasets'

    @property
    def folder_openalex(self):  # for intermediate files matched to openalex
        return self.raw_folder / 'openalex'

    def generate_datasets(self) -> Generator[Dataset, None, None]:
        files = list(self.folder_datasets.glob('*.jsonl'))
        logger.info(f'Searching for jsonl dumps in {self.folder_datasets}')
        logger.info(f'Found {len(files)} files')
        for file in files:
            yield read_file(file, key=f'{self.BASE}-{file.stem}')

    def prepare_datasets(self) -> None:
        pass

    def fetch_collection(self):
        logger.info('Fetching SIGIR2017 collection index')

        # Ensure target folder exists
        self.folder_datasets.mkdir(parents=True, exist_ok=True)

        # Load IDs
        df_rev = self.reviews
        df_cit = self.citations

        for rev, items in df_cit.groupby('document_id'):
            logger.info(f'Retrieving records for {rev}')
            review_info = df_rev.loc[rev]
            name = ID_MATCH.findall(review_info['url'])[0][0]
            file_name = self.folder_datasets / f'{name}.jsonl'
            logger.debug(f'  > {name} @ {file_name}')

            if file_name.exists():
                logger.info(f'Skipping {name} which already exists in folder {self.folder_datasets }')
                continue

            try:
                with open(file_name, 'w') as f_out:
                    for batch in batched(items.to_dict(orient='records'), batch_size=50):
                        batch_ids = [
                            {'pubmed_id': rec['url'][35:]}
                            for rec in batch
                        ]
                        lookup = {rec['url'][35:]: rec for rec in batch}
                        for record in fetch(ids=batch_ids):
                            f_out.write(json.dumps(record | lookup[record['pubmed_id']]) + '\n')
            except Exception as e:
                logger.error(f'Error fetching {name} from {rev}: {e}')
                logger.exception(e)
                file_name.unlink(missing_ok=True)


def read_file(file_path: Path, key: str) -> Dataset:
    with open(file_path, 'r') as f:
        records = [json.loads(line) for line in f]
        return Dataset(
            key=key,
            labels=[int(rec['included']) for rec in records],
            texts=[(rec['title'] or '') + ' ' + (rec['abstract'] or '') for rec in records]
        )


def read_sigir_dataset(key: str) -> Dataset:
    base = SIGIRCollection.BASE
    base_dir = settings.raw_data_path / 'datasets' / base
    base_name = key[len(base) + 1:]
    file_path = base_dir / f'{base_name}.jsonl'
    if not file_path.exists():
        raise AssertionError(f'Files for {key} not valid: {file_path}')

    return read_file(file_path, key)


if __name__ == '__main__':
    print("In module products __package__, __name__ ==", __package__, __name__)
    __package__ = 'loaders.sigir2017'
    SIGIRCollection().fetch_collection()
