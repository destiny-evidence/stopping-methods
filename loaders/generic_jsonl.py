from pathlib import Path

from shared.collection import AbstractCollection
from shared.config import settings
from shared.dataset import Dataset, Record


def read_file(file_path: Path, key: str) -> Dataset:
    with open(file_path, 'r') as f:
        records = [Record.model_validate_json(line) for line in f]
        records = [
            rec for rec in records
            if rec.abstract and len(rec.abstract) > 0
        ]
        return Dataset(key=key,
                       labels=[rec.label_abs for rec in records],
                       texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in records])


class GenericCollection(AbstractCollection):
    BASE: str = 'generic-jsonl'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        for file in self.raw_folder.glob('*.jsonl'):
            yield read_file(file, f'{self.BASE}-{file.stem}')


def read_jsonl_dataset(key: str) -> Dataset:
    base = GenericCollection.BASE
    base_dir = settings.raw_data_path / base
    base_name = key[len(base)+1:]
    file_path = base_dir / f'{base_name}.jsonl'
    if not file_path.exists():
        raise AssertionError(f'Files for {key} not valid: {file_path}')

    return read_file(file_path, key)
