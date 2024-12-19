from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class HeuristicFractionParamset(TypedDict):
    fractions: float


class HeuristicFractionLogEntry(AbstractLogEntry):
    key: str = 'HEURISTICFRACTION'


class HeuristicFraction(AbstractMethod):
    KEY = 'HEURISTICFRACTION'

    def parameter_options(self) -> Generator[HeuristicFractionParamset, None, None]:
        for target in [.01, .05, .057, .1, 0.2]:
            yield HeuristicFractionParamset(fractions=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                fractions: float) -> HeuristicFractionLogEntry:
        num_to_stop = self.dataset.n_total * fractions
        last_labels = list_of_labels[-num_to_stop:]

        return HeuristicFractionLogEntry(safe_to_stop=1 not in last_labels,
                                         num_seen=len(list_of_labels),
                                         num_included=list_of_labels.sum())
