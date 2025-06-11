from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class HeuristicFixedParamSet(TypedDict):
    num_to_stop: int


class HeuristicFixedLogEntry(AbstractLogEntry):
    KEY: str = 'HEURISTIC_FIX'
    num_to_stop: int


class HeuristicFixed(AbstractMethod):
    KEY: str = 'HEURISTIC_FIX'

    def parameter_options(self) -> Generator[HeuristicFixedParamSet, None, None]:
        for target in [50, 100, 200, 300]:
            yield HeuristicFixedParamSet(num_to_stop=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                num_to_stop: int=20) -> HeuristicFixedLogEntry:
        last_labels = list_of_labels[-min(len(list_of_labels), num_to_stop):]

        return HeuristicFixedLogEntry(safe_to_stop=1 not in last_labels,
                                      num_to_stop=num_to_stop)
