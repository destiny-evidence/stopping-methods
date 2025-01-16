from typing import Generator

from shared.collection import AbstractCollection
from shared.dataset import Dataset
from .synergy import SynergyDataset
from .generic_csv import GenericCollection as GenericCollectionCSV
from .generic_jsonl import GenericCollection as GenericCollectionJSON

# Collection = (SynergyDataset
#               | GenericCollectionCSV
#               | GenericCollectionJSON)

__all__ = ['SynergyDataset', 'GenericCollectionCSV', 'GenericCollectionJSON']


def it_collections() -> Generator[AbstractCollection, None, None]:
    """
    Iterates the implemented collections.
    :return:
    """
    for Collection in map(globals().get, __all__):
        yield Collection()


def it_datasets() -> Generator[Dataset, None, None]:
    for collection in it_collections():
        for dataset in collection.generate_datasets():
            yield dataset
