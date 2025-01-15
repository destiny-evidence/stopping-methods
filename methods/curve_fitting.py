from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class CurveFittingParamSet(TypedDict):
    recall_target: float
    curve: str


class CurveFittingLogEntry(AbstractLogEntry):
    KEY: str = 'curve_fitting'

    expected_includes: float
    expected_remaining: int
    predicted_recall: float


functions = {
    'exp': lambda x, a, b: a * np.exp(-b * x),  # aka Alison
}


class CurveFitting(AbstractMethod):
    KEY = 'CurveFitting'

    def parameter_options(self) -> Generator[CurveFittingParamSet, None, None]:
        for target in [.8, .9, .95, .99]:
            yield CurveFittingParamSet(recall_target=target, curve='exp')

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float,
                curve: str) -> CurveFittingLogEntry:
        labels = np.array(list_of_labels)

        # Fit the exponential curve and keep first parameter, which can be interpreted as
        # first one usually is the number of predicted includes
        params, _ = curve_fit(functions[curve],
                              np.arange(len(list_of_labels)),
                              labels.cumsum(),
                              maxfev=1000)
        a = params[0]
        n_seen_includes = labels.sum()  # the number of included records within seen data
        # Rescale the difference between number seen and number expected includes
        score = abs(a - n_seen_includes) / max(a, n_seen_includes)
        pred_recall = n_seen_includes / a  # the proportion of total predicted records that was already found

        # Alternative definition is to always normalise by expected includes
        # score = abs(a - labels.sum()) / a

        return CurveFittingLogEntry(safe_to_stop=score is not None and score < 1 - recall_target,
                                    expected_includes=a,
                                    predicted_recall=pred_recall,
                                    score=score,
                                    expected_remaining=int(abs(a - labels.sum())))
