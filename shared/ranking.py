import json
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from shared.dataset import Dataset


class TrainMode(str, Enum):
    RESET = "reset"
    FULL = "full"
    NEW = "new"


class AbstractRanker(ABC):
    def __init__(self,
                 train_mode: TrainMode,
                 tuning: bool = True,
                 **kwargs: dict[str, Any]):
        """

        :param dataset:
        :param train_on_new_only: Only use labels from latest batch for next training epoch
        :param train_from_scratch: Drop prior model and train from scratch for each batch
        :param retrain: if True, will train model from scratch after each batch
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.train_mode = train_mode
        self.tuning = tuning
        self.dataset = None

    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError()

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def train(self, idxs: list[int] | None = None, ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, idxs: list[int] | None = None, predict_on_all: bool = True) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _get_params(self, preview: bool = True) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        raise NotImplementedError()

    def attach_dataset(self, dataset: Dataset) -> None:
        self.clear()
        self.dataset = dataset

    def assembled_params(self, preview: bool = True) -> dict[str, Any]:
        return {
            'ranker': self.__class__.__name__,
            'train_mode': self.train_mode,
            'tuning': self.tuning,
            'dataset': self.dataset.KEY,
            'batch-strategy': self.dataset.batch_strategy,
            'batch-stat_batch_size': self.dataset.batch_size,
            'batch-dyn_min_batch_incl': self.dataset.min_batch_incl,
            'batch-dyn_min_batch_size': self.dataset.min_batch_size,
            'batch-dyn_growth_rate': self.dataset.growth_rate,
            'batch-dyn_max_batch_size': self.dataset.max_batch_size,
            'batch-inject_random_batch_every': self.dataset.inject_random_batch_every,
            **{f'model-{k}': v for k, v in self._get_params(preview=preview).items()},
        }

    def get_params(self, preview: bool = True) -> dict[str, Any]:
        return {
            'key': self.key,
            'name': self.name,
            **self.assembled_params(preview=preview),
        }

    def get_hash(self) -> str:
        return sha1(json.dumps(self.assembled_params(preview=True), sort_keys=True).encode('utf-8')).hexdigest()

    def store_info(self, target_path: Path, extra: dict[str, Any] | None = None) -> None:
        with open(target_path, 'w') as f:
            json.dump(self.get_params(preview=False) | (extra or {}), fp=f, indent=2)
