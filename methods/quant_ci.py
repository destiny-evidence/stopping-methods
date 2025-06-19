import logging
from typing import Generator, TypedDict
import numpy as np
import pandas as pd

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS
from shared.types import IntList, FloatList

logger = logging.getLogger('stop-quant')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# This is based on the implementation of the Quant Rule from the paper: Heuristic stopping rules for technology-assisted review (Yang 2021)
# Implementation taken from class QuantStoppingRule(StoppingRule) in Tarexo framework.
# https://github.com/eugene-yang/tarexp/blob/main/tarexp/component/stopping.py


class QuantCIParamSet(TypedDict):
    recall_target: float
    nstd: float


class QuantCILogEntry(AbstractLogEntry):
    KEY: str = 'QUANT_CI'
    recall_target: float
    nstd: float
    est_recall: float
    est_var: float | None = None


class QuantCI(AbstractMethod):
    KEY: str = 'QUANT_CI'

    def parameter_options(self) -> Generator[QuantCIParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            for nstd in [0, 1, 2]:
                yield QuantCIParamSet(recall_target=recall_target, nstd=nstd)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float = 0.9,
                nstd: float = 0) -> QuantCILogEntry:

        scores_all = np.array(list_of_model_scores)
        labels = np.array(list_of_labels)

        # mask nans and infs in scores
        mask = np.isfinite(scores_all)
        scores = scores_all[mask]
        labels = labels[mask[:len(labels)]]

        # return early if not enough labels are left
        if len(list_of_labels) < 50:
            return QuantCILogEntry(
                safe_to_stop=False,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=0.0
            )

        # calculate probability sums
        known_ps = scores[:len(labels)].sum()
        unknown_ps = scores[len(labels):].sum()

        est_recall = known_ps / (known_ps + unknown_ps) if (known_ps + unknown_ps) > 0 else 0

        if nstd == 0:
            # this is effectively the QUANT method (without CI)
            return QuantCILogEntry(
                safe_to_stop=est_recall >= recall_target,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=float(est_recall)
            )

        prod = scores * (1 - scores)
        all_var = prod.sum()
        unknown_var = prod[len(labels):].sum()
        est_var = ((known_ps ** 2 / (known_ps + unknown_ps) ** 4 * all_var) +
                   (1 / (known_ps + unknown_ps) ** 2 * (all_var - unknown_var)))
        safe_to_stop = (est_recall - nstd * np.sqrt(est_var)) >= recall_target

        return QuantCILogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            nstd=nstd,
            est_recall=float(est_recall),
            est_var=float(est_var)
        )


if __name__ == '__main__':
    import os
    import sys
    from shared.test import test_method, plots

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bs = 15
    dataset, results = test_method(QuantCI, QuantCIParamSet(recall_target=0.8, nstd=2),
                                   dataset_i=4, batch_size=bs)
    est_recalls = np.array([res['est_recall'] for res in results])

    fig, ax = plots(dataset, results)
    logger.debug(f'estimated recall: {[res['est_recall'] for res in results]}')
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax.plot(np.arange(len(results)) * bs, est_recalls * dataset.n_incl, label='est_recall*n_incl')
    ax2.plot(np.arange(len(results)) * bs, est_recalls, label='est_recall')
    # fig.savefig('data/plots/method_quant_viz.png')
    fig.show()
