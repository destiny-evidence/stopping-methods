import numpy as np
import pandas as pd

from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, StrList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class AlisonLogEntry(AbstractLogEntry):
    key: str = 'ALISON'


class Buscar(AbstractMethod):
    KEY = 'ALISON'

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series[bool] | pd.Series[int] | np.ndarray,
                num_total: int) -> AlisonLogEntry:
        score = 32

        return AlisonLogEntry(safe_to_stop=score is not None and score < 1 - confidence_level,
                              score=score,
                              num_seen=len(list_of_labels),
                              num_included=list_of_labels.sum())
