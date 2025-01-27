import logging
from copy import deepcopy
from pathlib import Path
from typing import Generator
from collections import defaultdict

import rispy

from shared.collection import AbstractCollection
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
            incl = file.stem.lower().endswith('_include')
            excl = file.stem.lower().endswith('_exclude')
            if not incl and not excl:
                logger.warning(f'Ignoring RIS file due to wrong naming convention: {file}')
                continue
            base = file.stem[:-8]
            pairs[base]['include' if incl else 'exclude'] = file

        logger.info(f'Found {len(pairs)} pairs')
        for pair, files in pairs.items():
            if len(files) != 2:
                logger.warning(f'Pair "{pair}" not valid, ignoring!')
                continue

            included = list(read_ris_file(files['include'], label_abs=True))
            excluded = list(read_ris_file(files['exclude'], label_abs=False, idx_offset=len(included)))

            yield Dataset(
                key=f'generic-paired-ris-{pair}',
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
