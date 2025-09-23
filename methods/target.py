from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import binom

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS, CONFIDENCE_TARGETS
from shared.types import IntList, FloatList, Indices, Mask


# TM QBCB
# https://dl.acm.org/doi/pdf/10.1145/3726302.3729879
# https://arxiv.org/pdf/2108.12746
# https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/qbcb.py

class TargetParamSet(TypedDict):
    target_recall: float
    confidence_level: float
    positive_sample_size: int


class TargetLogEntry(AbstractLogEntry):
    KEY: str = 'TM-QBCB'
    target_recall: float
    confidence_level: float
    positive_sample_size: int
    n_expected_incl: int | None = None


class Target(AbstractMethod):
    KEY: str = 'TM-QBCB'

    def parameter_options(self) -> Generator[TargetParamSet, None, None]:
        for tr in RECALL_TARGETS:
            for ct in CONFIDENCE_TARGETS:
                for ps in [5, 10, 25, 50, 100]:
                    yield TargetParamSet(target_recall=tr, confidence_level=ct, positive_sample_size=ps)

    @classmethod
    def compute(
            cls,
            dataset_size: int,
            list_of_labels: IntList,
            is_prioritised: Mask,
            list_of_model_scores: FloatList,
            positive_sample_size: int = 50,
            confidence_level: float = 0.95,
            target_recall: float = 0.95,
    ) -> TargetLogEntry:
        """
        QBCB from Lewis, Yang, and Frieder. CIKM '21.
        Certifying One-Phase Technology-Assisted Reviews.
        https://doi.org/10.1145/3459637.3482415
        
        Implementation inspired by https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/qbcb.py

        Important adjustments:
        The original target method makes assumptions about prior knowledge, either having a set of
        known relevant records from prior work or duplicating work by first annotating a random sample until
        a set number of relevant records are found. Depending on the variation of the target method, these known
        includes are hidden from the ranking model and one can stop when they (and others) are found again in the
        prioritised set.

        Here, we adjust the process by assuming the initial random sample is the reference. If not enough
        relevant records are included, it extends the sample (even from the prioritised set) until the configured
        number is met.
        """
        list_of_labels = np.array(list_of_labels)

        # Not enough data to meet the minimum size
        if len(list_of_labels) < positive_sample_size or list_of_labels.sum() < positive_sample_size:
            return TargetLogEntry(safe_to_stop=False, target_recall=target_recall,
                                  confidence_level=confidence_level, positive_sample_size=positive_sample_size)

        is_prioritised = np.array(is_prioritised)
        sample_random = list_of_labels[~is_prioritised]
        idxs_random = np.where(~is_prioritised)[0]
        if sample_random.sum() >= positive_sample_size:
            idxs_reference_sample = idxs_random[:idxs_random[np.where(sample_random.cumsum() >
                                                                      positive_sample_size)[0]][0]]
        else:
            # random sample not big enough. add more from prioritised set (in order) until criterion is met
            # FIXME: Maybe this should rather throw an error...
            idxs_prio = np.where(is_prioritised)[0]
            idxs_reference_sample = np.concat([
                idxs_random,
                idxs_prio[:idxs_prio[np.where(list_of_labels[is_prioritised].cumsum() >=
                                              (positive_sample_size - sample_random.sum()))[0]][0]]
            ])

        coeffs = binom.cdf(
            np.arange(positive_sample_size + 1),
            positive_sample_size,
            target_recall,
        )
        n_expected_incl = np.argmax(coeffs >= confidence_level) + 1
        idxs_reference_incl = idxs_reference_sample[list_of_labels[idxs_reference_sample] > 0]

        return TargetLogEntry(
            safe_to_stop=len(np.intersect1d(np.arange(len(list_of_labels)), idxs_reference_incl)) >= n_expected_incl,
            target_recall=target_recall,
            confidence_level=confidence_level,
            positive_sample_size=positive_sample_size,
            n_expected_incl=n_expected_incl,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots
    from matplotlib import pyplot as plt

    # plt.plot(binom.cdf(np.arange(11), 10, 0.8))
    # plt.show()

    dataset, results = test_method(Target,
                                   TargetParamSet(confidence_level=0.9, target_recall=0.9,
                                                  positive_sample_size=30), 2)
    fig, ax = plots(dataset, results)
    fig.show()
