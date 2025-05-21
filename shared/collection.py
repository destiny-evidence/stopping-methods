from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from shared.config import settings
from shared.dataset import Dataset


class AbstractCollection(ABC):
    # Name of the folder for this collection that is unique across all collections as a reference key
    BASE: str

    # NAME: str

    # TODO more default metadata

    @property
    def raw_folder(self) -> Path:
        return settings.raw_data_path / self.BASE

    @property
    def processed_folder(self) -> Path:
        return settings.processed_data_path / self.BASE

    @abstractmethod
    def fetch_collection(self) -> None:
        """
        Method to retrieve data from external sources and store it locally for further processing.
        Test if it exists first and do nothing if that's the case.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_datasets(self) -> None:
        """
        Read the raw data and write arrow files to the processed dataset folder
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_datasets(self) -> Generator[Dataset, None, None]:
        """
        Iterate through prepared data and yield Datasets
        :return:
        """
        raise NotImplementedError()
