import json
from abc import ABC, abstractmethod
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
                 dataset: Dataset,
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
        self.dataset = dataset
        self.train_mode = train_mode
        self.tuning = tuning

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError()

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, predict_on_all: bool = True) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _get_params(self, preview: bool = True) -> dict[str, Any]:
        raise NotImplementedError()

    def assembled_params(self, preview: bool = True) -> dict[str, Any]:
        return {
            'ranker': self.__class__.__name__,
            'train_mode': self.train_mode,
            'tuning': self.tuning,
            'dataset': self.dataset.KEY,
            'batch_strategy': self.dataset.batch_strategy,
            'stat_batch_size': self.dataset.batch_size,
            'dyn_min_batch_incl': self.dataset.min_batch_incl,
            'dyn_min_batch_size': self.dataset.min_batch_size,
            'dyn_growth_rate': self.dataset.growth_rate,
            'dyn_max_batch_size': self.dataset.max_batch_size,
            'inject_random_batch_every': self.dataset.inject_random_batch_every,
            **self._get_params(preview=preview)
        }

    def get_params(self, preview: bool = True) -> dict[str, Any]:
        return {
            'key': self.key,
            **self.assembled_params(preview=preview),
        }
    def get_hash(self) -> str:
        return sha1(json.dumps(self.assembled_params(preview=True), sort_keys=True).encode('utf-8')).hexdigest()

    def store_info(self, target_dir: Path, extra: dict[str, Any] | None = None) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / f'{self.key}.json', 'w') as f:
            json.dump({**self.get_params(preview=False), **(extra or {})}, fp=f, indent=2)
