import numpy as np

from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels, Scores
from typing import Generator


class MethodParams(_MethodParams):
    recall_target: float
    sample_size: float
    bias: float


class LogEntry(_LogEntry, MethodParams):
    est_incl: int


def calculate_ap(pid2label, ranked_pids, cutoff=0.5):
    num_rel = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_pids):
        label = pid2label[pid]
        if label >= cutoff:
            num_rel += 1
            total_precision += num_rel / (i + 1.0)

    return (total_precision / num_rel) if num_rel > 0 else 0.0


def calculate_losser(recall_cost, cost, N, R):
    return (1 - recall_cost) ** 2 + (100 / N) ** 2 * (cost / (R + 100)) ** 2


class SCAL(Method[Scores, None, None, None]):
    KEY: str = 'S-CAL'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for batch_size in [500, 1000, 2000]:
            yield MethodParams(batch_size=batch_size, threshold=threshold)

    @classmethod
    def compute(
            cls,
            *args,
            n_total: int,
            labels: Labels,
            scores: Scores,
            recall_target: float = 0.95,
            sample_size: float = 0.8,
            bias: float = 1.2,
            **kwargs,
    ) -> LogEntry:
        """
        Implements S-CAL
        > Cormack, CIKM 2016. "Scalability of Continuous Active Learning for Reliable High-Recall Text Classification"
        > via  https://dl.acm.org/doi/abs/10.1145/2983323.2983776

        Reference implementation:
        https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/scal.py
        """
        n_seen = len(labels)
        scores_seen = scores[:n_seen]

        # If currently seen dataset has no scores (random sample), skip
        if np.isnan(scores_seen).sum() >= n_seen:
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=False,
                recall_target=recall_target,
                sample_size=sample_size,
                confidence_level=None,
                est_incl=0,
                score=None,
                bias=bias,
            )

        # prio_scores = np.nan_to_num(scores_seen, nan=1)  # [~np.isnan(scores_seen)]
        prio_scores = scores_seen.copy()
        prio_scores[np.isnan(scores_seen)] = labels[np.isnan(scores_seen)]
        prio_labels = labels  # [~np.isnan(scores_seen)]

        sample_size_abs = int(sample_size * n_seen)  # N
        batch_size = 1  # B
        R = 0  # R^hat

        idxs = np.arange(len(prio_scores))
        idxs_sample = np.random.choice(idxs, sample_size_abs, replace=False)
        while len(idxs_sample) > 0:
            highest_ranking = idxs_sample[np.argsort(-scores_seen[idxs_sample])][:batch_size]

            if R == 1 or batch_size <= sample_size_abs:
                sub_size = batch_size
            else:
                sub_size = sample_size_abs

            idxs_sub = np.random.choice(highest_ranking, min(sub_size, len(highest_ranking)), replace=False)
            idxs_sample = np.array(list(set(idxs_sample) - set(idxs_sub)))

            R += (prio_labels[idxs_sub].sum() * batch_size) / sub_size

            batch_size += int(np.ceil(batch_size / 10))

        prevalence = ((1.05 * R) / n_total) * bias
        est_incl = prevalence * n_total

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=est_incl * recall_target < labels.sum(),
            recall_target=recall_target,
            sample_size=sample_size,
            confidence_level=None,
            est_incl=est_incl,
            score=labels.sum() / est_incl,
            bias=bias,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = MethodParams(recall_target=0.9, sample_size=0.9, bias=1.25)
    dataset, results = test_method(SCAL, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
