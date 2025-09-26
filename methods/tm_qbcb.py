from typing import Generator

import numpy as np
from scipy.stats import binom

from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS, CONFIDENCE_TARGETS
from shared.types import Labels


class MethodParams(_MethodParams):
    recall_target: float
    confidence_level: float
    positive_sample_size: int


class LogEntry(_LogEntry, MethodParams):
    n_sample: int
    required_overlap: int | None
    n_overlap: int | None


class TargetQBCB(Method[None, Labels, None, None]):
    KEY: str = 'TM_QBCB'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for tr in RECALL_TARGETS:
            for ct in CONFIDENCE_TARGETS:
                for ps in [5, 10, 25, 50, 100]:
                    yield MethodParams(recall_target=tr, confidence_level=ct, positive_sample_size=ps)

    @classmethod
    def compute(
            cls,
            n_total: int,  # Total number of records in datasets (seen + unseen)
            labels: Labels,
            full_labels: Labels,
            positive_sample_size: int = 50,
            confidence_level: float = 0.95,
            recall_target: float = 0.95,
            scores: None = None,
            is_prioritised: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements target method with quantile binomial confidence bound
        > Lewis, Yang, and Frieder. CIKM 2021. "Certifying One-Phase Technology-Assisted Reviews"
        > via https://dl.acm.org/doi/pdf/10.1145/3726302.3729879
        > via https://doi.org/10.1145/3459637.3482415
        > via https://arxiv.org/pdf/2108.12746
        
        Reference implementation
        https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/qbcb.py
        """
        y_seen = np.array(labels)
        y_all = np.array(full_labels)

        n_seen = len(y_seen)
        n_total = len(y_all)
        n_total_incl = y_all.sum()

        # Not enough data to meet the minimum size
        if n_total_incl < positive_sample_size:
            return LogEntry(KEY=cls.KEY,
                            safe_to_stop=False, recall_target=recall_target, n_sample=n_total,
                            confidence_level=confidence_level, positive_sample_size=positive_sample_size,
                            n_overlap=None, required_overlap=None, score=None)

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
        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=n_overlap >= required_overlap,
            recall_target=recall_target,
            confidence_level=confidence_level,
            positive_sample_size=positive_sample_size,
            required_overlap=required_overlap,
            n_overlap=n_overlap,
            n_sample=len(idxs_sample),
            score=n_overlap / (required_overlap + 1e-8),
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = MethodParams(confidence_level=0.8, recall_target=0.7, positive_sample_size=10)
    dataset, results = test_method(TargetQBCB, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
