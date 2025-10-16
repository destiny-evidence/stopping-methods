from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    stopping_percentage = 1.0, stopping_recall = None, target_recall = 1.0,  # autostop parameters
    sub_percentage = 0.8, bound_bt = 30, max_or_min = 'min', bucket_type = 'samplerel', ita = 1.05,  # scal parameters


class LogEntry(_LogEntry, MethodParams):
    pass


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
    return (1-recall_cost)**2 + (100/N)**2 * (cost/(R+100))**2
class SCAL(Method[None, None, None, None]):
    KEY: str = 'S-CAL'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for batch_size in [500, 1000, 2000]:
            yield MethodParams(batch_size=batch_size, threshold=threshold)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            batch_size: int = 1000,
            threshold: float = 0.1,
            scores: None = None,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements S-CAL
        > Cormack, CIKM 2016. "Scalability of Continuous Active Learning for Reliable High-Recall Text Classification"
        > via  https://dl.acm.org/doi/abs/10.1145/2983323.2983776

        Reference implementation:
        https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/scal.py
        """

        # TODO
        raise NotImplemented
        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=False,
            score=1.0,
            confidence_level=None,
            recall_target=None,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = MethodParams()
    dataset, results = test_method(SCAL, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
