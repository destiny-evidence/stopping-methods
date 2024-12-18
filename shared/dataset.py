import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator
from importlib.machinery import SourceFileLoader

from pydantic import BaseModel
import pandas as pd

from shared.config import settings


class Dataset(BaseModel):

    def get_labels(self) -> list[int] | pd.Series[int]:
        # TODO
        pass

    def get_texts(self) -> list[str] | pd.Series[str]:
        # TODO
        pass


class AbstractCollection(BaseModel, ABC):
    # Name of the folder for this collection that is unique across all collections as a reference key
    BASE: str

    NAME: str

    # TODO more default metadata

    @property
    def raw_folder(self) -> Path:
        return settings.raw_data_path / self.BASE

    @property
    def processed_folder(self) -> Path:
        return settings.processed_data_path / self.BASE

    @abstractmethod
    def fetch_collection(self):
        """
        Method to retrieve data from external sources and store it locally for further processing.
        Test if it exists first and do nothing if that's the case.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_datasets(self):
        """
        Read the raw data and write arrow files to the processed dataset folder
        :return:
        """
        raise NotImplementedError()

    def generate_datasets(self) -> Generator[Dataset, None, None]:
        """
        Iterate through prepared data and yield Datasets
        :return:
        """
        raise NotImplementedError()


def generate_collections() -> Generator[AbstractCollection, None, None]:
    """
    Iterates the implemented collections.
    :return:
    """
    for file in Path('datasets').iterdir():
        if file.suffix != '.py' or file.stem == '__init__':
            continue
        foo = SourceFileLoader("datasets", file).load_module()
        yield foo.Collection()
