from shared.collection import AbstractCollection
from shared.dataset import Dataset, Record


class GenericCollection(AbstractCollection):
    BASE: str = 'generic-jsonl'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        for file in self.raw_folder.glob('*.jsonl'):
            with open(file, 'r') as f:
                records = [Record.model_validate_json(line) for line in f]
                yield Dataset(key=f'generic-jsonl-{file.stem}',
                              labels=[rec.label_abs for rec in records],
                              texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in records])
