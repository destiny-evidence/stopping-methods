import logging
from typing import Generator

from shared.collection import AbstractCollection
from shared.dataset import Dataset
from .synergy import SynergyDataset
from .generic_csv import GenericCollection as GenericCollectionCSV, read_csv_dataset
from .generic_jsonl import GenericCollection as GenericCollectionJSON, read_jsonl_dataset
from .generic_paired_ris import GenericPairedRISCollection, read_paired_ris_dataset

logger = logging.getLogger('loaders')

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


def read_dataset(key: str) -> Dataset:
    if key.startswith(GenericPairedRISCollection.BASE):
        return read_paired_ris_dataset(key)
    if key.startswith(GenericCollectionCSV.BASE):
        return read_csv_dataset(key)
    if key.startswith(GenericCollectionJSON.BASE):
        return read_jsonl_dataset(key)

    raise ValueError(f'Dataset key {key} not supported')
