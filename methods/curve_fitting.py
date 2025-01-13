from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class CurveFittingParamset(TypedDict):
    recall_target: float


class CurveFittingLogEntry(AbstractLogEntry):
    KEY: str = 'CurveFitting'

    expected_includes: float
    expected_remaining: int
    predicted_recall: float


functions = {
    'exp': lambda x, a, b: a * np.exp(-b * x),

}


def exp_func(x, a, b):
    return


class CurveFitting(AbstractMethod):
    KEY = 'CurveFitting'

    def parameter_options(self) -> Generator[CurveFittingParamset, None, None]:
        for target in [.8, .9, .95, .99]:
            yield CurveFittingParamset(recall_target=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float) -> CurveFittingLogEntry:
        labels = np.array(list_of_labels)

        # Fit the exponential curve and keep first parameter, which can be interpreted as
        (a, _), _ = curve_fit(exp_func,
                              np.arange(len(list_of_labels)),
                              labels.cumsum(),
                              maxfev=1000)
        # Rescale the difference between number seen and number expected includes
        score = abs(a - labels.sum()) / max(a, labels.sum())
        my_seen_data = self.dataset.get_seen_data()  # a df showing the 'screened' data at each simulation step
        n_seen_includes = my_seen_data['labels'].sum()  # the number of included records within seen data
        pred_recall = n_seen_includes / a  # the proportion of total predicted records that was already found

        # a is the number of predicted includes

        # Alternative definition is to always normalise by expected includes
        # score = abs(a - labels.sum()) / a

        return CurveFittingLogEntry(safe_to_stop=score is not None and score < 1 - recall_target,
                              expected_includes=a,
                              predicted_recall=pred_recall,
                              score=score,
                              expected_remaining=int(abs(a - labels.sum())))
