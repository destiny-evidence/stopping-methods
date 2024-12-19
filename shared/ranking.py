from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from shared.dataset import Dataset


class AbstractRanker(ABC):
    KEY: str

    def __init__(self,
                 dataset: Dataset,
                 train_on_new_only: bool = False,
                 train_from_scratch: bool = True,
                 **kwargs: dict[str, Any]):
        """

        :param dataset:
        :param train_on_new_only: Only use labels from latest batch for next training epoch
        :param train_from_scratch: Drop prior model and train from scratch for each batch
        :param retrain: if True, will train model from scratch after each batch
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.train_on_new_only = train_on_new_only
        self.train_from_scratch = train_from_scratch

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self) -> np.ndarray:
        raise NotImplementedError()
