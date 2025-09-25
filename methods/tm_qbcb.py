from typing import Generator, TypedDict

import numpy as np
from scipy.stats import binom

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS, CONFIDENCE_TARGETS
from shared.types import IntList, FloatList, Mask


# TM QBCB
# https://dl.acm.org/doi/pdf/10.1145/3726302.3729879
# https://arxiv.org/pdf/2108.12746
# https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/qbcb.py

class TargetParamSet(TypedDict):
    recall_target: float
    confidence_level: float
    positive_sample_size: int


class TargetLogEntry(AbstractLogEntry):
    KEY: str = 'TM_QBCB'
    recall_target: float
    confidence_level: float
    positive_sample_size: int
    n_sample: int
    required_overlap: int | None = None
    n_overlap: int | None = None


class TargetQBCB(AbstractMethod):
    KEY: str = 'TM_QBCB'

    def parameter_options(self) -> Generator[TargetParamSet, None, None]:
        for tr in RECALL_TARGETS:
            for ct in CONFIDENCE_TARGETS:
                for ps in [5, 10, 25, 50, 100]:
                    yield TargetParamSet(recall_target=tr, confidence_level=ct, positive_sample_size=ps)

    def compute(
            self,
            dataset_size: int,
            list_of_labels: IntList,
            is_prioritised: Mask,
            list_of_model_scores: FloatList,
            positive_sample_size: int = 50,
            confidence_level: float = 0.95,
            recall_target: float = 0.95,
    ) -> TargetLogEntry:
        """
        QBCB from Lewis, Yang, and Frieder. CIKM '21.
        Certifying One-Phase Technology-Assisted Reviews.
        https://doi.org/10.1145/3459637.3482415
        
        Implementation inspired by https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/qbcb.py
        """
        y_seen = np.array(list_of_labels)
        y_all = np.array(self.dataset.labels)

        n_seen = len(y_seen)
        n_total = len(y_all)
        n_total_incl = y_all.sum()

        # Not enough data to meet the minimum size
        if n_total_incl < positive_sample_size:
            return TargetLogEntry(safe_to_stop=False, recall_target=recall_target, n_sample=n_total,
                                  confidence_level=confidence_level, positive_sample_size=positive_sample_size)

        idxs_seen = np.arange(n_seen)
        idxs_sample = np.arange(n_total)
        np.random.default_rng(4243).shuffle(idxs_sample)
        idxs_sample = idxs_sample[y_all[idxs_sample].cumsum() < positive_sample_size]
        idxs_sample_incl = idxs_sample[y_all[idxs_sample] > 0]

        coeffs = binom.cdf(
            np.arange(positive_sample_size + 1),
            positive_sample_size,
            recall_target,
        )
        required_overlap = np.argmax(coeffs >= confidence_level) + 1
        n_overlap = len(np.intersect1d(idxs_seen, idxs_sample_incl))
        return TargetLogEntry(
            safe_to_stop=n_overlap >= required_overlap,
            recall_target=recall_target,
            confidence_level=confidence_level,
            positive_sample_size=positive_sample_size,
            required_overlap=required_overlap,
            n_overlap=n_overlap,
            n_sample=len(idxs_sample),
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = TargetParamSet(confidence_level=0.8, recall_target=0.7,
                            positive_sample_size=10)
    dataset, results = test_method(TargetQBCB, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
