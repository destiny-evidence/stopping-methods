import json
import logging
from io import StringIO
from pathlib import Path
from typing import Generator, Any
import httpx
import pandas as pd
import numpy as np

from shared.collection import AbstractCollection
from shared.dataset import Dataset, Record

logger = logging.getLogger('synergy')


class SynergyDataset(AbstractCollection):
    BASE: str = 'SYNERGY'

    def generate_datasets(self) -> Generator[Dataset, None, None]:
        files = list(self.raw_folder.glob('*.jsonl'))
        logger.info(f'Searching for jsonl dumps in {self.raw_folder}')
        logger.info(f'Found {len(files)} files')
        for file in files:
            records = list(read_dataset(file))
            yield Dataset(
                key=f'synergy-{file.stem}',
                labels=[rec.label_abs for rec in records],
                texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in records]
            )

    def prepare_datasets(self) -> None:
        pass

    def fetch_collection(self):
        logger.info('Fetching Synergy collection index')
        index = httpx.get(
            'https://raw.githubusercontent.com/asreview/synergy-dataset/refs/heads/master/index.json').json()

        # Ensure target folder exists
        self.raw_folder.mkdir(parents=True, exist_ok=True)

        for name, info in index.items():
            if (self.raw_folder / f'{name}.jsonl').exists():
                logger.info(f'Skipping {name} which already exists on in folder')
                continue

            logger.info(f'Downloading {name}...')
            dataset = pd.read_csv(StringIO(httpx.get(info['url']).text))
            with open(self.raw_folder / f'{name}.jsonl', 'w') as out_file:
                for rec in dataset.to_dict('records'):
                    out_file.write(json.dumps(rec) + '\n')


def read_dataset(path: Path) -> Generator[Record, None, None]:
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            yield Record(
                id=safe_get(obj, 'record_id'),
                pubmed_id=safe_str(safe_get(obj, 'pubmedID')),
                title=safe_get(obj, 'title'),
                abstract=safe_get(obj, 'abstract'),
                doi=safe_get(obj, 'doi'),
                year=safe_get(obj, 'year'),
                # authors
                # label_abstract_screening
                label_abs=safe_get(obj, 'label_included'),
            )

def safe_str(val: int|None) -> str|None:
    if val is None:
        return None
    return str(val)
def safe_get(obj: dict[str, Any], field: str) -> str | int | bool | None:
    if field not in obj or pd.isnull(obj[field]):
        return None
    return obj[field]


def populate_ids(dataset: pd.DataFrame) -> Generator[dict[str, Any], None, None]:
    # for files like these:
    # https://github.com/asreview/synergy-dataset/blob/master/datasets/Appenzeller-Herzog_2019/Appenzeller-Herzog_2019_ids.csv

    fields = ','.join(FIELDS_TO_FETCH)
    for idx, row in dataset.iterrows():
        if row['openalex_id'] is None or len(row['openalex_id'].strip()) == 0:
            logger.warning(f'Skipping row {idx} with missing oa_id')
            continue
        try:
            work = httpx.get(f'{row['openalex_id']}?select={fields}').json()
            yield {
                **row.to_dict(),
                'openalex': work
            }
        except Exception as e:
            logger.warning(f'Failed to fetch for row {idx} at {row['openalex_id']}')
            logger.exception(e)


FIELDS_TO_FETCH = [
    'id',
    'doi',
    'title',
    'display_name',
    'publication_year',
    'publication_date',
    'ids',
    'language',
    'primary_location',
    'type',
    'type_crossref',
    'indexed_in',
    'open_access',
    'authorships',
    'institution_assertions',
    'countries_distinct_count',
    'institutions_distinct_count',
    'corresponding_author_ids',
    'corresponding_institution_ids',
    'apc_list',
    'apc_paid',
    'fwci',
    'has_fulltext',
    'fulltext_origin',
    # 'cited_by_count',
    # 'citation_normalized_percentile',
    # 'cited_by_percentile_year',
    # 'biblio',
    'is_retracted',
    'is_paratext',
    'primary_topic',
    # 'topics',
    'keywords',
    # 'concepts',
    # 'mesh',
    'locations_count',
    'locations',
    'best_oa_location',
    # 'sustainable_development_goals',
    'grants',
    'datasets',
    'versions',
    # 'referenced_works_count',
    'referenced_works',
    # 'related_works',
    'abstract_inverted_index',
    # 'cited_by_api_url',
    # 'counts_by_year',
    'updated_date',
    'created_date'
]
