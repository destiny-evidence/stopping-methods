from typing import Generator, TypedDict
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

"""
Adapted from https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/knee_kneedle.py
"""

class RhoMode(str, Enum): 
    DYNAMIC = "DYNAMIC"   # compute rho dynamically from data
    STATIC = "STATIC"     # use a static rho value

class KneeParamSet(TypedDict):   # parameters for one run of the algorithm (I think)
    stopping_beta: int
    mode: RhoMode
    rho_target: float
    n_windows: int

class KneeLogEntry(AbstractLogEntry): 
    KEY: str = 'KNEE'
    mode: RhoMode
    stopping_beta: int
    rho_target: float
    rho: float
    n_windows: int
    knees: list[int] | None = None

def detect_knee(data: np.ndarray, window_size: int = 1, s: float = 10) -> list[int]:
    """
    Detect the so-called knee in the data.

    The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle). 
    Adapted from https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/knee_kneedle.py

    @param data: The 2d data to find an knee in.
    @param window_size: The data is smoothed using Gaussian kernel average smoother, this parameter is the window used for averaging (higher values mean more smoothing, try 3 to begin with).
    @param s: How many "flat" points to require before we consider it a knee.
    @return: The knee values.
    """
    assert len(data.shape) == 2                    # guard: 2-D only
    data_size = data.shape[0]
    if data_size == 1:                             # trivial case
        return []

    # smooth
    smoothed_data = []
    for i in range(data_size):
        start_index = max(i - window_size, 0)
        end_index   = min(i + window_size, data_size - 1)

        sum_x_weight = sum_y_weight = sum_index_weight = 0.0
        for j in range(start_index, end_index):
            index_weight     = norm.pdf(abs(j - i) / window_size, 0, 1)
            sum_index_weight += index_weight
            sum_x_weight     += index_weight * data[j][0]
            sum_y_weight     += index_weight * data[j][1]

        if sum_index_weight > 0:                   # weighted mean
            smoothed_data.append((
                sum_x_weight / sum_index_weight,
                sum_y_weight / sum_index_weight
            ))
        else:                                      
            smoothed_data.append((data[i][0], data[i][1]))

    smoothed_data = np.array(smoothed_data)

    # normalize
    normalized_data = MinMaxScaler().fit_transform(smoothed_data)

    # difference
    differed_data = np.array([(x, y - x) for x, y in normalized_data])

    # find indices for local maximums
    candidate_indices = []
    for i in range(1, data_size - 1):
        if (
                differed_data[i][1] > differed_data[i - 1][1]
                and
                differed_data[i][1] > differed_data[i + 1][1]
        ):
            candidate_indices.append(i)

    # threshold - how flat does it need to be to be considered a knee
    step = s * (normalized_data[-1][0] - data[0][0]) / (data_size - 1)

    # knees
    knee_indices = []
    for i, candidate_index in enumerate(candidate_indices):
        if i + 1 < len(candidate_indices):  # not last second
            end_index = candidate_indices[i + 1]
        else:
            end_index = data_size

        threshold = differed_data[candidate_index][1] - step

        for j in range(candidate_index, end_index):
            if differed_data[j][1] < threshold:
                knee_indices.append(candidate_index)
                break

    return knee_indices

class Knee(AbstractMethod):
    KEY: str = 'KNEE'

    def parameter_options(self) -> Generator[KneeParamSet, None, None]:
        for mode in [RhoMode.DYNAMIC, RhoMode.STATIC]:
            for stopping_beta in [100, 1000]:
                for rho_target in [3, 5, 6, 8, 10]:
                    for n_windows in [10, 100, 1000]:
                        yield KneeParamSet(
                            mode=mode,
                            stopping_beta=stopping_beta,
                            rho_target=rho_target,
                            n_windows=n_windows
                        )

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,            
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                mode: RhoMode = RhoMode.DYNAMIC,
                stopping_beta: int = 100,
                rho_target: float = 6,
                n_windows: int = 100) -> KneeLogEntry:
        
        if len(list_of_labels) < stopping_beta:             # beta stopping condition (not enough screened, i.e., < beta)
            return KneeLogEntry(safe_to_stop=False,
                                mode=mode,
                                stopping_beta=stopping_beta,
                                rho_target=rho_target,
                                rho=0.0,
                                n_windows=n_windows)
        
        batch_size = max(1, int(self.dataset.n_total / n_windows))    # dont run knee every iteration? 

        labels = np.array(list_of_labels)
        if labels.sum() == 0:                                # no positives yet so knee wouldn't make sense
            return KneeLogEntry(safe_to_stop=False,
                                mode=mode,
                                stopping_beta=stopping_beta,
                                rho_target=rho_target,
                                rho=0.0,
                                n_windows=n_windows)
        
        # inclusion curve where x-axis is "proportion of collection" and y-axis is "recall" - make knee detection not sensitive to dataset size
        data = np.array([np.arange(len(labels)) / self.dataset.n_total,
                         labels.cumsum() / labels.sum()]).T
        data = data[::batch_size]                           
        actual_n_windows = len(data)
        if actual_n_windows < 2:
            return KneeLogEntry(safe_to_stop=False,
                                    mode=mode,
                                    stopping_beta=stopping_beta,
                                    rho_target=rho_target,
                                    rho=0.0,
                                    n_windows=actual_n_windows)

        knee_indices = detect_knee(data)
        if not knee_indices:
            return KneeLogEntry(safe_to_stop=False,
                                    mode=mode,
                                    stopping_beta=stopping_beta,
                                    rho_target=rho_target,
                                    rho=0.0,
                                    n_windows=actual_n_windows,
                                    knees=[])
        
        # rho calculation from last knee
        last_knee = knee_indices[-1]
        r1   = data[last_knee][1] * labels.sum()            # de-normalise, atm inclusion curve’s x-axis is “proportion of collection” and y-axis is “recall”, so must undo that scaling before computing ρ.
        rank1 = data[last_knee][0] * self.dataset.n_total
        r2   = data[-1][1] * labels.sum()
        rank2 = data[-1][0] * self.dataset.n_total
        try:
            current_rho = (r1 / rank1) / ((r2 - r1 + 1) / (rank2 - rank1))
        except: 
            print(
                "(rank1, r1) = ({} {}), (rank2, r2) = ({} {})".format(
                    rank1, r1, rank2, r2
                )
            )
            current_rho = 0.0

        effective_rho_target = (156 - min(labels.sum(), 150)) if mode == RhoMode.DYNAMIC else rho_target
        safe_to_stop = current_rho >= effective_rho_target and len(labels) >= stopping_beta

        return KneeLogEntry(safe_to_stop=safe_to_stop,
                                mode=mode,
                                stopping_beta=stopping_beta,
                                rho_target=effective_rho_target,
                                rho=current_rho,
                                n_windows=actual_n_windows,
                                knees=[int(data[i][0] * self.dataset.n_total) for i in knee_indices])
        






