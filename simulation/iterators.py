from typing import Generator

from shared.dataset import AbstractCollection, Dataset
import datasets
import methods
import rankings
from shared.method import AbstractMethod


def it_collections() -> Generator[AbstractCollection, None, None]:
    """
    Iterates the implemented collections.
    :return:
    """
    for Collection in datasets.__all__:
        yield Collection()


def it_datasets() -> Generator[Dataset, None, None]:
    for collection in it_collections():
        for dataset in collection.generate_datasets():
            yield dataset


def it_rankers(dataset:Dataset) -> Generator[AbstractCollection, None, None]:
    for Ranker in rankings.__all__:
        yield Ranker(dataset)


def it_methods(dataset: Dataset) -> Generator[AbstractMethod, None, None]:
    for Method in methods.__all__:
        yield Method(dataset)
