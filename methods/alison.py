from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class AlisonParamset(TypedDict):
    recall_target: float


class AlisonLogEntry(AbstractLogEntry):
    KEY: str = 'ALISON'

    expected_includes: float
    expected_remaining: int
    predicted_recall: float


def exp_func(x, A, lamda):
    return A * (1 - np.exp(-lamda * x))


class Alison(AbstractMethod):
    KEY = 'ALISON'

    def parameter_options(self) -> Generator[AlisonParamset, None, None]:
        for target in [.8, .9, .95, .99]:
            yield AlisonParamset(recall_target=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float) -> AlisonLogEntry:
        labels = np.array(list_of_labels)

        try:
            (a, _), _ = curve_fit(exp_func,
                                  np.arange(len(list_of_labels)),
                                  labels.cumsum(),
                                  maxfev=10000)
        except:
            a=self.dataset.n_total#the method might fail to fit and throw an error. If this happens, we set the number of expected includes to the total number of documents in the dataset (ie. worst-case scenario)

        # Rescale the difference between number seen and number expected includes
        score = abs(a - labels.sum()) / max(a, labels.sum())
        my_seen_data = self.dataset.get_seen_data()  # a df showing the 'screened' data at each simulation step
        n_seen_includes = my_seen_data['labels'].sum()  # the number of included records within seen data
        pred_recall = n_seen_includes / a  # the proportion of total predicted records that was already found

        # a is the number of predicted includes

        # Alternative definition is to always normalise by expected includes
        # score = abs(a - labels.sum()) / a

        return AlisonLogEntry(safe_to_stop=score is not None and score < 1 - recall_target,
                              expected_includes=a,
                              predicted_recall=pred_recall,
                              score=score,
                              expected_remaining=int(abs(a - labels.sum())))
