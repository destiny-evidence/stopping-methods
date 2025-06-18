import logging
from pathlib import Path
from typing import Generator, TypedDict
import numpy as np
import pandas as pd

from shared.dataset import RankedDataset
from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS
from shared.types import IntList, FloatList

logger = logging.getLogger('stop-quant')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# This is based on the implementation of the Quant Rule from the paper: Heuristic stopping rules for technology-assisted review (Yang 2021)
# Implementation taken from class QuantStoppingRule(StoppingRule) in Tarexo framework.
# https://github.com/eugene-yang/tarexp/blob/main/tarexp/component/stopping.py


class QuantParamSet(TypedDict):
    recall_target: float
    nstd: float


class QuantLogEntry(AbstractLogEntry):
    KEY: str = 'QUANT'
    recall_target: float
    nstd: float
    est_recall: float
    est_var: float | None = None


class Quant(AbstractMethod):
    KEY: str = 'QUANT'

    def parameter_options(self) -> Generator[QuantParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            for nstd in [0, 1, 2]:
                yield QuantParamSet(recall_target=recall_target, nstd=nstd)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float = 0.9,
                nstd: float = 0) -> QuantLogEntry:

        if len(list_of_labels) < 2:
            return QuantLogEntry(
                safe_to_stop=False,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=0.0
            )

        scores = np.array(list_of_model_scores)
        labels = np.array(list_of_labels)

        # `ps` stands for probability sum
        known_ps = scores[:len(labels)].sum()
        unknown_ps = scores[len(labels):].sum()
        est_recall = known_ps / (known_ps + unknown_ps)

        if nstd == 0:
            return QuantLogEntry(
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
        safe_to_stop = est_recall - nstd * np.sqrt(est_var) >= recall_target
        return QuantLogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            nstd=nstd,
            est_recall=float(est_recall),
            est_var=float(est_var)
        )



if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(Quant, QuantParamSet(recall_target=0.95, nstd=1), 3)
    fig, ax = plots(dataset, results)
    fig.show()