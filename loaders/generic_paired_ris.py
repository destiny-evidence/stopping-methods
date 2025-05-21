import logging
from copy import deepcopy
from pathlib import Path
from typing import Generator
from collections import defaultdict

import rispy

from shared.collection import AbstractCollection
from shared.config import settings
from shared.dataset import Dataset, Record

logger = logging.getLogger(__name__)


class GenericPairedRISCollection(AbstractCollection):
    BASE: str = 'generic-paired-ris'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        files = list(self.raw_folder.glob('*.ris'))
        logger.info(f'Searching for pairs of RIS files in {self.raw_folder}')
        logger.info(f'Found {len(files)} files')

        pairs = defaultdict(dict)
        for file in files:
            incl = file.stem.lower().endswith('_includes')
            excl = file.stem.lower().endswith('_excludes')
            if not incl and not excl:
                logger.warning(f'Ignoring RIS file due to wrong naming convention: {file}')
                continue
            base = file.stem[:-9]
            pairs[base]['include' if incl else 'exclude'] = file

        logger.info(f'Found {len(pairs)} pairs')
        for pair, files in pairs.items():
            if len(files) != 2:
                logger.warning(f'Pair "{pair}" not valid, ignoring!')
                continue

            included = list(read_ris_file(files['include'], label_abs=True))
            excluded = list(read_ris_file(files['exclude'], label_abs=False, idx_offset=len(included)))

            yield Dataset(
                key=f'{self.BASE}-{pair}',
                labels=[rec.label_abs for rec in included + excluded],
                texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in included + excluded]
            )


def read_paired_ris_dataset(key: str) -> Dataset:
    base = GenericPairedRISCollection.BASE
    base_dir = settings.raw_data_path / base
    base_name = key[len(base) + 1:]
    file_incl = base_dir / f'{base_name}_INCLUDES.ris'
    file_excl = base_dir / f'{base_name}_EXCLUDES.ris'
    if not (file_incl.exists() and file_excl.exists()):
        raise AssertionError(f'Files for {key} not valid: \n'
                             f'{file_incl}\n'
                             f'{file_excl}')

    included = list(read_ris_file(file_incl, label_abs=True))
    excluded = list(read_ris_file(file_excl, label_abs=False, idx_offset=len(included)))

    return Dataset(
        key=key,
        labels=[rec.label_abs for rec in included + excluded],
        texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in included + excluded]
    )


def read_ris_file(filepath: Path,
                  label_abs: bool,
                  label_ft: bool | None = None,
                  idx_offset: int = 0,
                  extra_fields: dict[str, str] | None = None) -> Generator[Record, None, None]:
    mapping = deepcopy(rispy.TAG_KEY_MAPPING)
    for src, tgt in (extra_fields or {}).items():
        mapping[src] = tgt  # e.g. U1->pmid, C8->OAid

    entries = rispy.load(filepath, mapping=mapping)

    for idx, entry in enumerate(entries, start=idx_offset):
        yield Record(
            id=idx,
            doi=entry.get('doi'),
            pubmed_id=entry.get('pmid'),
            openalex_id=entry.get('OAid'),
            title=entry.get('title', entry.get('primary_title')),
            abstract=entry.get('abstract'),
            keywords='; '.join(entry['keywords']) if 'keywords' in entry else None,
            label_abs=label_abs,
            label_ft=label_ft,
        )
