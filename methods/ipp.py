from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import curve_fit

from shared.method import AbstractMethod, AbstractLogEntry, NUM_WINDOWS, CONFIDENCE_TARGETS, RECALL_TARGETS
from shared.types import IntList, FloatList


# Implementation based on:
# https://github.com/ReemBinHezam/TAR_Stopping_Point_Processes
# https://github.com/levnikmyskin/salt/blob/main/baselines/sneyd_stevenson/ipp.py
# https://github.com/alisonsneyd/stopping_criteria_counting_processes/blob/main/run_stopping_point_experiments.ipynb
# via https://dl.acm.org/doi/abs/10.1145/3631990

class PointProcessParamSet(TypedDict):
    n_windows: int
    target_recall: float
    confidence_level: float


class PointProcessLogEntry(AbstractLogEntry):
    KEY: str = 'IPP'
    n_windows: int
    target_recall: float
    confidence_level: float
    est_incl: int | None = None


def model_func_power(x, a, k):
    return a * x ** k


class PointProcess(AbstractMethod):
    KEY: str = 'IPP'

    def parameter_options(self) -> Generator[PointProcessParamSet, None, None]:
        for nw in NUM_WINDOWS:
            for cl in CONFIDENCE_TARGETS:
                for tr in RECALL_TARGETS:
                    yield PointProcessParamSet(n_windows=nw, target_recall=tr, confidence_level=cl)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                n_windows: int = 10,
                target_recall: float = 0.95,
                confidence_level: float = 0.95,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None) -> PointProcessLogEntry:

        n_seen = len(list_of_labels)
        n_incl_seen = list_of_labels.sum()

        try:
            # Set up window bounds
            window_size = n_seen // n_windows
            indices = np.arange(n_seen)
            windows = indices[: window_size * n_windows].reshape(-1, n_windows)
            window_edges = windows[:, [0, -1]]
            window_edges[0, 0] += 1
            list_of_labels = np.array(list_of_labels)

            # Example values:
            # dataset_size = 100
            # list_of_labels = [0,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0]
            # n_windows = 5            |  window_size = 4    | n_incl_seen = 9
            # windows:                 |  window_edges:
            # [[ 0,  1,  2,  3,  4],   |     [[ 1,  9],
            #  [ 5,  6,  7,  8,  9],   |      [10, 19]]
            #  [10, 11, 12, 13, 14],   |
            #  [15, 16, 17, 18, 19]]   |

            # Find power curve points
            idxs_windows = window_edges[:, 0]
            inclusion_rates = list_of_labels[windows].sum(axis=1) / window_size
            # idxs_windows:    [ 1,  5, 10, 15]
            # inclusion_rates: [ 4,  1,  2,  2] / 4

            p0 = [0.1, 0.001]
            (a, k), pcov = curve_fit(model_func_power, idxs_windows, inclusion_rates, p0)
            est_incl_likelihood = model_func_power(np.arange(1, n_seen + 1), a, k)
            est_incl_seen = round(np.sum(est_incl_likelihood))
            # est_incl_seen: 9

            if n_incl_seen >= target_recall * est_incl_seen:
                mu = (a / (k + 1)) * (dataset_size ** (k + 1) - 1)
                est_incl = np.argmin(poisson.cdf(np.arange(dataset_size) + 1, mu) < confidence_level) + 1
                # mu = 26.05102
                # est_incl = 35
                return PointProcessLogEntry(safe_to_stop=(target_recall * est_incl) <= n_incl_seen,
                                            n_windows=n_windows,
                                            confidence_level=confidence_level,
                                            target_recall=target_recall,
                                            score=n_incl_seen / (target_recall * est_incl),
                                            est_incl=est_incl)
        except:
            pass
        return PointProcessLogEntry(safe_to_stop=False,
                                    n_windows=n_windows,
                                    confidence_level=confidence_level,
                                    target_recall=target_recall)


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = PointProcessParamSet(
        target_recall=0.95,
        confidence_level=0.95,
        n_windows=20,
    )
    dataset, results = test_method(PointProcess, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
