from typing import Generator

from shared.collection import AbstractCollection
from shared.dataset import Dataset
import datasets
import methods
import rankings
from shared.method import AbstractMethod
from shared.ranking import AbstractRanker


def it_collections() -> Generator[AbstractCollection, None, None]:
    """
    Iterates the implemented collections.
    :return:
    """
    for Collection in map(datasets.__dict__.get, datasets.__all__):
        yield Collection()


def it_datasets() -> Generator[Dataset, None, None]:
    for collection in it_collections():
        for dataset in collection.generate_datasets():
            yield dataset


def it_rankers(dataset: Dataset) -> Generator[AbstractRanker, None, None]:
    for Ranker in map(rankings.__dict__.get, rankings.__all__):
        yield Ranker(dataset)


def it_methods(dataset: Dataset) -> Generator[AbstractMethod, None, None]:
    for Method in map(methods.__dict__.get, methods.__all__):
        yield Method(dataset)
