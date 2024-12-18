from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, StrList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class AlisonParamset(TypedDict):
    recall_target: float


class AlisonLogEntry(AbstractLogEntry):
    key: str = 'ALISON'

    expected_includes: float
    expected_remaining: int


def exp_func(x, a, b):
    return a * np.exp(-b * x)


class Alison(AbstractMethod):
    KEY = 'ALISON'

    def parameter_options(self) -> Generator[AlisonParamset, None, None]:
        for target in [.8, .9, .95, .99]:
            yield AlisonParamset(recall_target=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series[bool] | pd.Series[int] | np.ndarray,
                num_total: int,
                recall_target: float) -> AlisonLogEntry:
        labels = np.array(list_of_labels)

        # Fit the exponential curve and keep first parameter, which can be interpreted as
        (a, _), _ = curve_fit(exp_func,
                              np.arange(len(list_of_labels)),
                              labels.cumsum(),
                              maxfev=1000)
        # Rescale the difference between number seen and number expected includes
        score = abs(a - labels.sum()) / max(a, labels.sum())
        # Alternative definition is to always normalise by expected includes
        # score = abs(a - labels.sum()) / a

        return AlisonLogEntry(safe_to_stop=score is not None and score < 1 - recall_target,
                              num_seen=len(list_of_labels),
                              num_included=list_of_labels.sum(),
                              expected_includes=a,
                              score=score,
                              expected_remaining=abs(a - labels.sum()))
