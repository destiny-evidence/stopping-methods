from typing import Generator

from shared.dataset import AbstractCollection
import datasets


def generate_collections() -> Generator[AbstractCollection, None, None]:
    """
    Iterates the implemented collections.
    :return:
    """
    for Collection in datasets.__all__:
        yield Collection()
