from typing import Generator

from shared.collection import AbstractCollection
from shared.dataset import Dataset


class SynergyDataset(AbstractCollection):
    KEY: str = 'SYNERGY'
    BASE: str = 'SYNERGY'

    def generate_datasets(self) -> Generator[Dataset, None, None]:
        yield from ()

    def fetch_collection(self):
        pass

    def prepare_datasets(self):
        pass
