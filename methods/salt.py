from typing import Generator

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine

from shared.types import Bounds, Scores, Labels, BinaryScores, Indices
from shared.method import Method, _MethodParams, _LogEntry, CONFIDENCE_TARGETS, RECALL_TARGETS


class MethodParams(_MethodParams):
    alpha: float
    nstd: int
    use_adjusted: bool
    use_margin: bool
    confidence_level: float
    recall_target: float


class LogEntry(_LogEntry, MethodParams):
    est_recall: float | None
    margin_recall: float | None


class SALT(Method[Scores, None, Bounds, None]):
    KEY: str = 'SALτ'

    def parameter_options(self) -> Generator[MethodParams, None, None]:
        for cl in CONFIDENCE_TARGETS:
            for tr in RECALL_TARGETS:
                for alpha in [0.1, 0.3, 0.5, 0.8, 1.0]:
                    for adjusted in [True, False]:
                        # https://github.com/levnikmyskin/salt/blob/d965b75edee76f0eea17408892eb0a716f217b69/sld/sld_stopping.py#L24
                        # SALτ paper terminology:
                        #  - SALtQuantCI has nstd = 2;
                        yield MethodParams(alpha=alpha, nstd=2, use_adjusted=adjusted, use_margin=False,
                                           confidence_level=cl, recall_target=tr)
                        #  - SALt has nstd = 0, use_margin = False;
                        yield MethodParams(alpha=alpha, nstd=0, use_adjusted=adjusted, use_margin=False,
                                           confidence_level=cl, recall_target=tr)
                        #  - SAL^r_t has nstd = 0, use_margin = True;
                        yield MethodParams(alpha=alpha, nstd=0, use_adjusted=adjusted, use_margin=True,
                                           confidence_level=cl, recall_target=tr)

                    # for nstd in [0, 1, 2]:
                    #     for adjusted in [True, False]:
                    #         for margin in [True, False]:
                    #             yield MethodParams(alpha=alpha, nstd=nstd, use_adjusted=adjusted, use_margin=margin,
                    #                    confidence_level=cl, recall_target=tr)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            scores: Scores,
            bounds: Bounds,
            batch_size: int = 1000,
            threshold: float = 0.1,
            nstd: int = 0,
            alpha: float = 0.3,
            use_adjusted: bool = True,
            use_margin: bool = False,
            confidence_level: float = 0.95,
            recall_target: float = 0.95,
            is_prioritised: None = None,
            full_labels: None = None,
    ) -> LogEntry:
        """
        Implements SALτ
        > Molinari and Esuli, DMKD 2024. "SALτ: efficiently stopping TAR by improving priors estimates"
        > via  https://link.springer.com/article/10.1007/s10618-023-00961-5

        Reference implementation
        https://github.com/levnikmyskin/salt
        """

        if (len(bounds) < 2 and use_adjusted) or (len(bounds) < 5 and not use_adjusted):
            # Need at least a few batches to compute this rule; return early otherwise
            return LogEntry(KEY=cls.KEY, safe_to_stop=False,
                            use_adjusted=use_adjusted, use_margin=use_margin, alpha=alpha, nstd=nstd,
                            confidence_level=confidence_level, recall_target=recall_target,
                            score=None, est_recall=None, margin_recall=None)
        scores = np.nan_to_num(scores)
        scores: BinaryScores = np.array([1 - scores, scores]).T

        idxs = np.arange(n_total)
        idxs_batch = idxs[bounds[-1][0]:bounds[-1][1]]
        idxs_seen = idxs[:bounds[-1][0]]
        idxs_unseen = idxs[len(labels):]

        if use_adjusted:
            _, scores_adjusted = sld(
                scores[idxs_seen],  # FIXME `scores` should actually be scores before retraining
                scores[idxs_unseen],  # FIXME `scores` should actually be scores before retraining
                adjust=True,
                trust=1 - cosine(scores[idxs_batch, 1],  # FIXME `scores` should actually be scores before retraining
                                 scores[idxs_batch, 1]),
            )
            sc = np.copy(scores[:, 1])
            sc[idxs_unseen] = scores_adjusted[:, 1]
            error = (np.sum(np.abs(labels[idxs_batch].mean() - sc[idxs_batch].mean())) /
                     (2 * (1 - labels[idxs_batch].mean())))

            if error > alpha:
                sc = np.copy(scores[:, 1])
        else:
            _, scores_adjusted = sld(scores[idxs_unseen],
                                     scores[idxs_unseen])  # FIXME `scores` should actually be scores before retraining
            sc = np.copy(scores[:, 1])
            sc[idxs_unseen] = scores_adjusted[:, 1]

        recall = quant_stop(probs=sc, idxs_seen=idxs_seen, idxs_unseen=idxs_unseen, idxs_batch=idxs_batch, nstd=nstd)
        margin_target_recall = (2 * recall_target - recall_target ** 2) if use_margin else recall_target

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=recall >= margin_target_recall,
            score=recall / margin_target_recall,
            use_adjusted=use_adjusted, use_margin=use_margin, alpha=alpha, nstd=nstd,
            confidence_level=confidence_level, recall_target=recall_target,
            est_recall=recall, margin_recall=margin_target_recall,
        )


def sld(
        scores_seen: BinaryScores, scores_unseen: BinaryScores,
        epsilon=1e-6, max_steps: int = 1000,
        adjust: bool = False, trust: float = 0.5, tau: float = 1.,
):
    """
    Implements the prior correction method based on EM

    Originally presented here:
    > Saerens, Latinne and Decaestecker, 2002
    > "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    > via https://www.isys.ucl.ac.be/staff/marco/Publications/Saerens2002a.pdf

    This implementation follows Algorithm 3 for `adjust=False` or Algorithm 4 for `adjust=True` in
    > Molinari and Esuli, DMKD 2024. "SALτ: efficiently stopping TAR by improving priors estimates"
    > via  https://link.springer.com/article/10.1007/s10618-023-00961-5

    Implementation based on
    https://github.com/levnikmyskin/salt/blob/d965b75edee76f0eea17408892eb0a716f217b69/sld/sld.py#L19-L86
    """
    priors = scores_seen.mean(0)
    posteriors = scores_unseen

    priors_step = priors
    posteriors_step = posteriors

    for _ in range(max_steps):
        ratios = (priors_step + 1e-10) / (priors + 1e-10)

        if adjust:
            ratios = -((trust * (-ratios + 1)) ** tau) + 1

        denominators = (ratios * posteriors).sum()
        posteriors_step = ratios * posteriors / denominators

        if posteriors_step.shape[0] == 0:
            break

        priors_step_ = priors  # remember previous steps' priors
        priors_step = posteriors_step.mean(axis=0)

        if np.abs(priors_step_ - priors_step).sum() < epsilon:
            break
    return priors_step, posteriors_step


def quant_stop(probs: Scores,
               idxs_seen: Indices, idxs_unseen: Indices, idxs_batch: Indices,
               nstd: int) -> float:
    tr_probs = probs[idxs_seen]
    val_probs = probs[idxs_unseen]
    te_probs = probs[idxs_batch]
    known_sum = tr_probs.sum() + val_probs.sum()
    est_recall = known_sum / (known_sum + te_probs.sum())
    if nstd == 0:
        return est_recall
    prod = probs * (1 - probs)
    all_var = prod.sum()
    unknown_var = prod[idxs_batch].sum()

    est_var = (
            (known_sum ** 2 / (known_sum + te_probs.sum()) ** 4 * all_var) +
            (1 / (known_sum + te_probs.sum()) ** 2 * (all_var - unknown_var))
    )
    return est_recall - nstd * np.sqrt(est_var)


if __name__ == '__main__':
    from shared.test import test_method, plots

    #  - SALtQuantCI has nstd = 2;
    params = MethodParams(alpha=1.0, nstd=2, use_adjusted=True, use_margin=False,
                          confidence_level=.9, recall_target=.9)
    dataset, results = test_method(SALT, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()

    # - SALt has nstd = 0, use_margin = False;
    params = MethodParams(alpha=1.0, nstd=0, use_adjusted=True, use_margin=False,
                          confidence_level=.9, recall_target=.9)
    dataset, results = test_method(SALT, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()

    #  - SAL^r_t has nstd = 0, use_margin = True;
    params = MethodParams(alpha=1.0, nstd=0, use_adjusted=True, use_margin=True,
                          confidence_level=.9, recall_target=.9)
    dataset, results = test_method(SALT, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
