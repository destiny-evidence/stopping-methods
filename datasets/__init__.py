import logging
from typing import Generator

from shared.collection import AbstractCollection
from shared.dataset import Dataset
from .synergy import SynergyDataset
from .generic_csv import GenericCollection as GenericCollectionCSV
from .generic_jsonl import GenericCollection as GenericCollectionJSON
from .generic_paired_ris import GenericPairedRISCollection

logger = logging.getLogger('datasets')

# Collection = (SynergyDataset
#               | GenericCollectionCSV
#               | GenericCollectionJSON)

# 'SynergyDataset',
__all__ = ['GenericCollectionCSV', 'GenericCollectionJSON', 'GenericPairedRISCollection']


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


def prepare_collections() -> None:
    for collection in it_collections():
        logger.info(f'Fetching {collection.BASE}')
        collection.fetch_collection()
        logger.info(f'Preparing {collection.BASE}')
        collection.prepare_datasets()
