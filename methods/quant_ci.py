import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class QuantCIParamSet(TypedDict):
    recall_target: float
    nstd: float


class QuantCILogEntry(AbstractLogEntry):
    KEY: str = 'QUANT'
    recall_target: float
    nstd: float
    est_recall: float
    est_var: float | None = None


class QuantCI(AbstractMethod):
    KEY: str = 'QUANT'

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

        if len(list_of_labels) < 2:
            return QuantCILogEntry(
                safe_to_stop=False,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=0.0
            )

        scores_all = np.array(list_of_model_scores)
        labels = np.array(list_of_labels)

        # mask nans and infs in scores
        mask = ~np.isfinite(scores_all)
        scores = np.ma.array(scores_all, mask=mask)

        # `ps` stands for probability sum
        known_ps = scores[:len(labels)].sum()
        unknown_ps = scores[len(labels):].sum()

        if known_ps is np.ma.masked or unknown_ps is np.ma.masked:
            est_recall = np.nan
        elif (known_ps + unknown_ps) <= 0:
            est_recall = 0
        else:
            est_recall = known_ps / (known_ps + unknown_ps)

        if nstd == 0:
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
        if est_var is np.ma.masked:
            est_var = np.nan
            safe_to_stop = False
        else:
            safe_to_stop = est_recall - nstd * np.sqrt(est_var) >= recall_target
        
        return QuantCILogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            nstd=nstd,
            est_recall=float(est_recall),
            est_var=float(est_var)
        )


def test(ranking: Path,
         batch_size: int = 25):
    dataset = RankedDataset(ranking)
    batches, labels, scores, is_prioritised = dataset.data
    labels = np.array(labels)
    data = np.array([np.arange(len(labels)), labels.cumsum()]).T
    # data = MinMaxScaler().fit_transform(data)
    recall_target = 0.9
    stopper = QuantCI(dataset)
    xs = []
    log = []
    for n_seen in range(0, dataset.n_total, batch_size):
        logger.info(f'Running batch for items {n_seen:,}–{n_seen + batch_size:,}')

        batch_labels = labels[n_seen:n_seen + batch_size]
        score = stopper.compute(
            list_of_labels=batch_labels,
            list_of_model_scores=np.array(scores),
            is_prioritised=is_prioritised[n_seen:n_seen + batch_size],
            recall_target=recall_target,
            nstd=0.1,
        )
        logger.info(f' > score: {score}')
        log.append(score)
        xs.append(n_seen + batch_size)

    fig, ax = plt.subplots()
    ax.plot(data[:, 0] / dataset.n_total, data[:, 1], label='include', ls='-')
    ax2 = ax.twinx()
    ax2.plot(np.array(xs) / dataset.n_total, [l.est_var for l in log], label='est_var')
    ax2.plot(np.array(xs) / dataset.n_total, [l.est_recall for l in log], label='est_recall')
    # ax2.hlines(recall_target, 0, 1, lw=1)
    fig.legend()
    fig.show()



if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(QuantCI, QuantCIParamSet(recall_target=0.8, nstd=2), 4)
    fig, ax = plots(dataset, results)
    fig.savefig('data/plots/method_quant_viz.png')
    logger.debug(f'estimated recall: {[res['est_recall'] for res in results]}')

# if __name__ == '__main__':
#     import typer
#     from matplotlib import pyplot as plt
#     from sklearn.preprocessing import MinMaxScaler
#
#     typer.run(test)
#
#     # print(data.shape)
#     # knees = detect_knee(data, window_size=1, s=1)
#     # data = np.array([np.arange(len(labels)) / len(labels),
#     #                  labels.cumsum() / labels.sum()]).T[:3000]
#     # knees = detect_knee(data[::500], window_size=1, s=100)
