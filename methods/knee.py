import logging
from typing import Generator, TypedDict
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class KneeParamSet(TypedDict):
    window_size: int
    s: int
    n_windows: float


class KneeLogEntry(AbstractLogEntry):
    KEY: str = 'KNEE'
    window_size: int
    s: int
    n_windows: int
    knees: list[int] | None = None


class Knee(AbstractMethod):
    KEY: str = 'KNEE'

    def parameter_options(self) -> Generator[KneeParamSet, None, None]:
        for ws in [1, 3, 10]:
            for s in [5, 10, 20]:
                for nw in [10, 100, 1000]:
                    yield KneeParamSet(window_size=ws, s=s, n_windows=nw)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                window_size: int,
                s: int,
                n_windows: int) -> KneeLogEntry:
        """
           Detect the so-called knee in the data.

           The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle).
           via https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/knee.py

           // @param data: The 2d data to find a knee in.
           @param window_size: The data is smoothed using Gaussian kernel average smoother, this parameter is the
                               window used for averaging (higher values mean more smoothing, try 3 to begin with).
           @param s: How many "flat" points to require before we consider it a knee.
           @param batch_scale: proportional batch size; originally not in the implementation, but we give it the
                               full list of all annotations and the algorithm benefits from a more coarse resolution.
                               0.05 seems to be a reasonable factor
           @return: The knee values.
           """
        batch_size = max(1, int(self.dataset.n_total / n_windows))

        labels = np.array(list_of_labels)
        # Transform list of labels to inclusion curve;
        # normalise x to fraction of dataset total and normalise y to (0,1)
        data = np.array([np.arange(len(list_of_labels)) / self.dataset.n_total,
                         labels.cumsum() / labels.sum()]).T
        # Create fake batches and select only every Nth entry from the curve
        data = data[::batch_size]
        n_windows = len(data)

        if n_windows < 2:
            return KneeLogEntry(safe_to_stop=False,
                                s=s, n_windows=n_windows, window_size=window_size)

        # smooth
        smoothed_data = []
        for i in range(n_windows):
            if 0 < i - window_size:
                start_index = i - window_size
            else:
                start_index = 0
            if n_windows - i - window_size <= 0:
                end_index = n_windows - 1
            else:
                end_index = i + window_size

            sum_x_weight = 0
            sum_y_weight = 0
            sum_index_weight = 0
            for j in range(start_index, end_index):
                index_weight = norm.pdf(abs(j - i) / window_size, 0, 1)
                sum_index_weight += index_weight
                sum_x_weight += index_weight * data[j][0]
                sum_y_weight += index_weight * data[j][1]

            smoothed_x = sum_x_weight / sum_index_weight
            smoothed_y = sum_y_weight / sum_index_weight

            smoothed_data.append((smoothed_x, smoothed_y))

        smoothed_data = np.array(smoothed_data)

        # normalize
        normalized_data = MinMaxScaler().fit_transform(smoothed_data)

        # difference
        differed_data = np.array([(x, y - x) for x, y in normalized_data])

        # find indices for local maximums
        candidate_indices = []
        for i in range(1, n_windows - 1):
            if (
                    differed_data[i][1] > differed_data[i - 1][1]
                    and
                    differed_data[i][1] > differed_data[i + 1][1]
            ):
                candidate_indices.append(i)

        # threshold
        step = s * (normalized_data[-1][0] - data[0][0]) / self.dataset.n_total

        # knees
        knee_indices = []
        for i, candidate_index in enumerate(candidate_indices):
            if i + 1 < len(candidate_indices):  # not last second
                end_index = candidate_indices[i + 1]
            else:
                end_index = n_windows

            threshold = differed_data[candidate_index][1] - step

            for j in range(candidate_index, end_index):
                if differed_data[j][1] < threshold:
                    knee_indices.append(candidate_index)
                    break

        return KneeLogEntry(safe_to_stop=len(knee_indices) > 0,
                            s=s, n_windows=n_windows, window_size=window_size,
                            knees=[ki * batch_size for ki in knee_indices])

# def alison_knee():
#     # https://github.com/alisonsneyd/poisson_stopping_method/blob/master/run_stopping_point_algorithms.py
#     sample_props = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
#                     0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  # proportion of docs to sample
#     min_rel_in_sample = 20  # min number rel docs must be initial sample to proceed with algorithm
#     n_windows = 10  # number of windows to male from sample
#     des_prob = 0.95  # set minimum desired probability for poisson
#     des_recall = 0.7  # set minimum desired recall level
#     target_size = 10  # set size target set for target method (Cormack and Grossman set to 10)
#     knee_rho = 6  # knee method rho (Cormack and Grossman set to 6)
#
#     # get batches as defined by autonomous tar method (Cormack and Grossman, 2015)
#     def get_batches(n_docs):
#
#         a = 1
#         batches = []
#
#         while a < n_docs:
#             batches.append(a)
#             a += math.ceil(a / 10)
#
#         return batches
#
#     # fn to find the knee i for a given value of s
#     def find_knee(rel_list, s):
#
#         n_rel = np.sum(rel_list[0:s])
#         m = n_rel / s  # slope of line connecting (s, n_rel) to origin
#
#         distances = []
#         for x_r in range(1, s + 1):  # x-values = rank
#             y_r = np.sum(rel_list[0:x_r])  # get corresponding y-value on gain curve
#             distance = (abs(m * x_r - y_r)) / math.sqrt(m ** 2 + 1)  # standard formula dist point to line
#             distances.append(distance)
#
#         max_distance_rank = distances.index(max(distances)) + 1  # get rank of max distance
#
#         return max_distance_rank
#
#     # fn to calculate the slope ratio given i and s
#     def get_slope_ratio(rel_list, i, s):  # get rho pg 5 Cormack and Grossman, i < s
#
#         big_num = (np.sum(rel_list[0:i])) / i
#         big_denom = (1 + np.sum(rel_list[i:s])) / (s - i)
#         rho = big_num / big_denom
#
#         return rho
#
#     # fn to get the stopping point for the knee method
#     def get_knee_stopping_point_var_adjust(rel_list, batches, target_ratio, low_rel_adjust):
#
#         idx = 0
#         rho = 0
#         adjusted_ratio = low_rel_adjust + target_ratio
#
#         while (rho < adjusted_ratio) and (idx < len(batches)):
#
#             s = batches[idx]
#             i = find_knee(rel_list, s)
#
#             relret = np.sum(rel_list[0:i])
#             adjusted_ratio = low_rel_adjust + target_ratio - min(relret, low_rel_adjust)
#
#             if i < s:
#                 rho = get_slope_ratio(rel_list, i, s)
#             else:
#                 rho = 0
#
#             idx += 1
#
#         return (i, s)  # knee of stopping point, stopping point
#
#     batches = get_batches(n_docs)
#
#     knee, knee_stop = get_knee_stopping_point_var_adjust(rel_list, batches, knee_rho, 150)[0:2]
#     knee_recall = calc_recall(rel_list, knee_stop)
#     knee_effort = knee_stop
#     knee_accept = calc_accept(knee_recall, des_recall)
#     score_dic[query_id].append((knee_recall, knee_effort, knee_accept))
#
#     knee, knee_stop = get_knee_stopping_point_var_adjust(rel_list, batches, knee_rho, 50)[0:2]
#     knee_recall = calc_recall(rel_list, knee_stop)
#     knee_effort = knee_stop
#     knee_accept = calc_accept(knee_recall, des_recall)
#     score_dic[query_id].append((knee_recall, knee_effort, knee_accept))
