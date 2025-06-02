from typing import Generator, TypedDict
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

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
           https://raghavan.usc.edu/papers/kneedle-simplex11.pdf

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
        # normalise x to fraction of dataset total and normalise y to (0, 1)

        if labels.sum() == 0:
            return KneeLogEntry(safe_to_stop=False,
                                s=s, n_windows=n_windows, window_size=window_size)

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


if __name__ == '__main__':
    from shared.test import test_method, plots
    dataset, results = test_method(Knee, KneeParamSet(window_size=20, s=10, n_windows=100), 2 )
    fig, ax = plots(dataset, results)
    fig.show()