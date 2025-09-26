from typing import Generator
import numpy as np
from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS
from shared.types import Labels, Scores


class MethodParams(_MethodParams):
    nstd: float
    recall_target: float


class LogEntry(_LogEntry, MethodParams):
    est_recall: float
    est_var: float | None


class QuantCI(Method[Scores, None, None, None]):
    KEY: str = 'QUANT_CI'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for recall_target in RECALL_TARGETS:
            # for nstd in [0, 1, 2]:
            for nstd in [1, 2]:  # n_std == 0 is equivalent to quant/heuristic_scores!
                yield MethodParams(recall_target=recall_target, nstd=nstd)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            scores: Scores,
            recall_target: float = 0.9,
            nstd: float = 0,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements QuantCI
        > Yang 2021. "Heuristic stopping rules for technology-assisted review"
        > via  https://dl.acm.org/doi/abs/10.1145/2983323.2983776

        Reference implementations:
        https://github.com/eugene-yang/tarexp/blob/main/tarexp/component/stopping.py
        https://github.com/levnikmyskin/salt/blob/main/baselines/lewis_yang/quant_ci.py
        """
        scores_all = np.array(scores)
        labels = np.array(labels)

        # mask nans and infs in scores
        mask = np.isfinite(scores_all)
        scores = scores_all[mask]
        labels = labels[mask[:len(labels)]]

        # return early if not enough labels are left
        if len(labels) < 50:  # FIXME: Where does the 50 come from?
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=False,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=0.0,
                confidence_level=None,
                score=None,
                est_var=None,
            )

        # calculate probability sums
        known_ps = scores[:len(labels)].sum()
        unknown_ps = scores[len(labels):].sum()

        est_recall = known_ps / (known_ps + unknown_ps) if (known_ps + unknown_ps) > 0 else 0

        if nstd == 0:
            # this is effectively the QUANT method (without CI)
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=est_recall >= recall_target,
                recall_target=recall_target,
                nstd=nstd,
                est_recall=float(est_recall),
                confidence_level=None,
                score=None,
                est_var=None,
            )

        prod = scores * (1 - scores)
        all_var = prod.sum()
        unknown_var = prod[len(labels):].sum()
        est_var = ((known_ps ** 2 / (known_ps + unknown_ps) ** 4 * all_var) +
                   (1 / (known_ps + unknown_ps) ** 2 * (all_var - unknown_var)))
        safe_to_stop = (est_recall - nstd * np.sqrt(est_var)) >= recall_target

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            nstd=nstd,
            est_recall=float(est_recall),
            est_var=float(est_var),
            confidence_level=None,
            score=None,
        )


if __name__ == '__main__':
    import os
    import sys
    import logging
    from shared.test import test_method, plots

    logger = logging.getLogger('stop-quant')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bs = 15
    params = MethodParams(recall_target=0.8, nstd=1)
    dataset, results = test_method(QuantCI, params, dataset_i=4, batch_size=bs)
    est_recalls = np.array([res['est_recall'] for res in results])

    fig, ax = plots(dataset, results, params)
    logger.debug(f'estimated recall: {[res['est_recall'] for res in results]}')
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax.plot(np.arange(len(results)) * bs, est_recalls * dataset.n_incl, label='est_recall*n_incl')
    ax2.plot(np.arange(len(results)) * bs, est_recalls, label='est_recall')
    # fig.savefig('data/plots/method_quant_viz.png')
    fig.show()
