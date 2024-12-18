from abc import ABC, abstractmethod, abstractclassmethod
from typing import Any, Generator

from pydantic import BaseModel
import pandas as pd
import numpy as np

from shared.dataset import Dataset
from shared.types import IntList, FloatList


class AbstractLogEntry(BaseModel):
    key: str
    safe_to_stop: bool
    num_seen: int
    num_included: int
    score: float | None = None


class AbstractMethod(BaseModel, ABC):
    KEY: str

    def __init__(self, dataset: Dataset, **kwargs: dict[str, Any]):
        super().__init__(**kwargs)
        self.dataset = dataset

    @abstractmethod
    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series[bool] | pd.Series[int] | np.ndarray,
                num_total: int,
                **kwargs: dict[str, Any]) -> AbstractLogEntry:
        raise NotImplementedError()

    def retrospective(self,
                      list_of_labels: IntList,
                      list_of_model_scores: FloatList,
                      is_prioritised: list[int] | list[bool] | pd.Series[bool] | pd.Series[int] | np.ndarray,
                      num_total: int,
                      batch_size: int = 100,
                      **kwargs: dict[str, Any]) -> Generator[AbstractLogEntry, None, None]:
        for n_seen_batch in range(batch_size, len(list_of_labels), batch_size):
            batch_labels = list_of_labels[:n_seen_batch]
            yield self.compute(list_of_labels=batch_labels,
                               list_of_model_scores=list_of_model_scores,
                               is_prioritised=is_prioritised,
                               num_total=num_total,
                               **kwargs)
