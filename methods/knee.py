from typing import Generator, TypedDict
from enum import StrEnum
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]

DEV_MODE = __name__ == '__main__'


class SmoothingMethod(StrEnum):
    GAUSS = 'gauss'
    SAVGOL = 'savgol'


class KneeParamSet(TypedDict):
    window_size: int
    polyorder: int
    threshold_ratio: float
    threshold_peak: float
    smoothing: SmoothingMethod


class KneeLogEntry(AbstractLogEntry):
    KEY: str = 'KNEE'
    window_size: int
    polyorder: int
    threshold_ratio: float
    threshold_peak: float
    slope_ratio: float
    smoothing: SmoothingMethod


class Knee(AbstractMethod):
    KEY: str = 'KNEE'

    def parameter_options(self) -> Generator[KneeParamSet, None, None]:
        # for window_size in [100, 200]:
        #     for th_r in [4, 6, 7, 10]:
        #         for th_p in [0.2, 0.3, 0.4]:
        #             yield KneeParamSet(window_size=window_size, threshold_ratio=th_r, threshold_peak=th_p,
        #                                polyorder=1, smoothing=SmoothingMethod.GAUSS)
        for window_size in [500]:
            for th_r in [2, 3, 4, 7]:
                for th_p in [0.2, 0.3, 0.4]:
                    yield KneeParamSet(window_size=window_size, threshold_ratio=th_r, threshold_peak=th_p,
                                       polyorder=1, smoothing=SmoothingMethod.SAVGOL)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                window_size: int = 500,
                smoothing: SmoothingMethod = SmoothingMethod.GAUSS,
                polyorder: int = 1,
                threshold_ratio: float = 6.0,
                threshold_peak: float = 0.3,
                ) -> KneeLogEntry:
        """
        Detect the so-called knee in the data.
        This is based on kneedle but adapted so that it actually has a chance to work.
          https://raghavan.usc.edu/papers/kneedle-simplex11.pdf
        First used for stopping:
          https://dl.acm.org/doi/pdf/10.1145/2911451.2911510

        Implemented idea here:
          1) Find the knee
             - smooth curve and norm to 0–1 range
             - compute diff between smooth curve to line (0,0)-(seen, included)
             - find argmax on diff -> knee
          2) Compute slopes before and after knee point
          3) Compute and test ratio of pre-/post-slopes
        """
        labels = np.array(list_of_labels)

        if labels.sum() == 0:
            return KneeLogEntry(safe_to_stop=False, window_size=window_size, polyorder=polyorder,
                                threshold_ratio=threshold_ratio, slope_ratio=0, smoothing=smoothing)

        x = np.arange(len(list_of_labels))
        x_norm = x / self.dataset.n_total
        curve = labels.cumsum() / labels.sum()  # rescales y-values of inclusion curve
        window_size_ = min(window_size, len(list_of_labels))

        if smoothing == SmoothingMethod.SAVGOL:
            curve_smooth = savgol_filter(curve, window_length=window_size_, polyorder=polyorder)
        elif smoothing == SmoothingMethod.GAUSS:
            curve_smooth = gaussian_filter1d(curve, sigma=window_size_)
        else:
            raise AttributeError(f'Unknown smoothing method {smoothing}')
        curve_smooth_norm = curve_smooth / curve_smooth.max()
        curve_smooth_norm = curve_smooth_norm - curve_smooth_norm.min()
        curve_smooth_norm = curve_smooth_norm / curve_smooth_norm.max()
        baseline = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        diff = curve_smooth_norm - baseline

        knee = np.argmax(diff)
        slope_pre = curve_smooth_norm[knee]
        slope_post = 1.0 - curve_smooth_norm[knee]
        # print((len(x) - knee), knee, len(x), slope_pre, slope_post, slope_pre / slope_post)
        if slope_post == 0 or (len(x) - knee) < 50 or ((len(x) - knee) / len(x)) < 0.05 or diff.max() < threshold_peak:
            return KneeLogEntry(safe_to_stop=False, window_size=window_size, polyorder=polyorder,
                                threshold_ratio=threshold_ratio, threshold_peak=threshold_peak,
                                slope_ratio=0, smoothing=smoothing)

        slope_ratio = slope_pre / slope_post

        if DEV_MODE and (len(list_of_labels) % 200) == 0 and len(list_of_labels) > 100:
            fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150, figsize=(12, 5))
            ax1.plot(x, curve, label='raw')
            ax1.plot(x, curve_smooth, label='smoothed')
            ax1.plot(x, curve_smooth_norm, label='normed')
            ax1.grid(lw=0.1, ls='--')

            ax2.plot(x_norm, baseline, lw=0.1, ls='--', label='base')
            ax2.plot(x_norm, curve_smooth_norm, label='curve')
            ax2.plot(x_norm, diff, label='diff')

            ax3 = ax2.twinx()
            d1 = np.gradient(diff)
            ax3.plot(d1, label='d1d', lw=0.2)
            d2 = np.gradient(d1)
            ax3.plot(d2, label='d2d', lw=0.2)
            ax3.plot(np.gradient(np.gradient(curve_smooth_norm)), label='d2', lw=0.2)

            ax2.grid(lw=0.1, ls='--')
            ax2.set_xlim(0, 1)
            # ax2.set_ylim(0, 1)
            ax2.text(0, 0.9, f'{curve_smooth_norm[knee]:.1f} -> {slope_ratio:.2f}')
            ax2.plot((slope_ratio > threshold_ratio).astype(int), label='stop')
            fig.legend(loc='outside upper right')
            fig.suptitle(f'x_norm.max()={x_norm.max():.2f} '
                         f'// window_size={window_size} '
                         f'// n_labels={len(list_of_labels):,}',
                         fontsize=12)
            fig.tight_layout()
            fig.show()

        return KneeLogEntry(safe_to_stop=slope_ratio > threshold_ratio,
                            polyorder=polyorder,
                            window_size=window_size,
                            slope_ratio=slope_ratio,
                            threshold_ratio=threshold_ratio,
                            threshold_peak=threshold_peak,
                            smoothing=smoothing)


if __name__ == '__main__':
    from shared.test import test_method, plots
    from matplotlib import pyplot as plt

    # DEV_MODE = False
    # dataset, results = test_method(Knee, KneeParamSet(window_size=100, polyorder=1,
    #                                                   threshold_ratio=5, smoothing=SmoothingMethod.GAUSS), 2)
    # plt.plot(np.linspace(0, 1, len(results)) * dataset.n_total, [r['slope_ratio'] for r in results])
    # plt.show()
    # dataset, results = test_method(Knee, KneeParamSet(window_size=200, polyorder=1,
    #                                                   threshold_ratio=5, smoothing=SmoothingMethod.GAUSS), 2)
    # plt.plot([r['slope_ratio'] for r in results])
    # plt.show()

    dataset, results = test_method(Knee, KneeParamSet(window_size=500, polyorder=1, threshold_peak=0.3,
                                                      threshold_ratio=3, smoothing=SmoothingMethod.SAVGOL), 5)
    plt.plot(np.arange(len(results)) / len(results) * dataset.n_total, [r['slope_ratio'] for r in results])
    plt.show()

    fig, ax = plots(dataset, results)
    fig.show()
