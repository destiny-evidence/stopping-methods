from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class HeuristicFractionParamSet(TypedDict):
    fraction: float


class HeuristicFractionLogEntry(AbstractLogEntry):
    KEY: str = 'HEURISTIC_FRAC'
    num_to_stop: int
    fraction: float


class HeuristicFraction(AbstractMethod):
    KEY: str = 'HEURISTIC_FRAC'

    def parameter_options(self) -> Generator[HeuristicFractionParamSet, None, None]:
        for target in [.01, .05, .075, .1, 0.2]:
            yield HeuristicFractionParamSet(fraction=target)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                fraction: float = 0.05) -> HeuristicFractionLogEntry:
        num_to_stop = int(dataset_size * fraction)
        last_labels = list_of_labels[-min(len(list_of_labels), num_to_stop):]

        return HeuristicFractionLogEntry(safe_to_stop=1 not in last_labels,
                                         num_to_stop=num_to_stop,
                                         fraction=fraction)
