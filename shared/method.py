from abc import ABC, abstractmethod
from typing import Any, Generator

from pydantic import BaseModel
import pandas as pd
import numpy as np

from shared.dataset import Dataset, RankedDataset
from shared.types import IntList, FloatList


class AbstractLogEntry(BaseModel):
    KEY: str
    safe_to_stop: bool
    score: float | None = None


RECALL_TARGETS = [.8, .9, .95, .99]
CONFIDENCE_TARGETS = [.8, .9, .95, .99]
INCLUSION_THRESHOLDS = [.25, .5, .75, .9]
WINDOW_SIZES = [50, 500, 1000]
NUM_WINDOWS = [5, 10, 20, 50, 100]


class AbstractMethod(ABC):
    KEY: str

    def __init__(self, dataset: Dataset | RankedDataset, **kwargs: dict[str, Any]):
        super().__init__(**kwargs)
        self.dataset = dataset

    @abstractmethod
    def parameter_options(self) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def compute(cls,
                dataset_size: int,  # Total number of records in datasets (seen + unseen)
                list_of_labels: IntList,
                list_of_model_scores: FloatList | None = None,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                **kwargs: dict[str, Any]) -> AbstractLogEntry:
        raise NotImplementedError()

    @classmethod
    def retrospective(cls,
                      dataset_size: int,  # Total number of records in datasets (seen + unseen)
                      list_of_labels: IntList,
                      list_of_model_scores: FloatList | None = None,
                      is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                      batch_size: int = 100,
                      **kwargs: dict[str, Any]) -> Generator[AbstractLogEntry, None, None]:
        for n_seen_batch in range(batch_size, len(list_of_labels), batch_size):
            batch_labels = list_of_labels[:n_seen_batch]
            yield cls.compute(dataset_size=dataset_size,
                              list_of_labels=batch_labels,
                              list_of_model_scores=list_of_model_scores,
                              is_prioritised=is_prioritised,
                              **kwargs)

    def compute_(self,
                 list_of_labels: IntList,
                 list_of_model_scores: FloatList | None = None,
                 is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                 **kwargs: dict[str, Any]) -> AbstractLogEntry:
        return self.compute(dataset_size=self.dataset.n_total,
                            list_of_labels=list_of_labels,
                            list_of_model_scores=list_of_model_scores,
                            is_prioritised=is_prioritised,
                            **kwargs)

    def retrospective_(self,
                       list_of_labels: IntList,
                       list_of_model_scores: FloatList | None = None,
                       is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                       batch_size: int = 100,
                       **kwargs: dict[str, Any]) -> Generator[AbstractLogEntry, None, None]:
        yield from self.retrospective(dataset_size=self.dataset.n_total,
                                      list_of_labels=list_of_labels,
                                      list_of_model_scores=list_of_model_scores,
                                      is_prioritised=is_prioritised,
                                      batch_size=batch_size,
                                      **kwargs)
