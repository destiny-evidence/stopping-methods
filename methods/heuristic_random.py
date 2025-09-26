from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS
from shared.types import Labels, Sampling
from typing import Generator


class MethodParams(_MethodParams):
    recall_target: float


class LogEntry(_LogEntry, MethodParams):
    est_incl: int


class HeuristicRandom(Method[None, None, None, Sampling]):
    KEY: str = 'HEURISTIC_RANDOM'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for recall_target in RECALL_TARGETS:
            yield MethodParams(recall_target=recall_target)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            is_prioritised: Sampling,
            batch_size: int = 1000,
            threshold: float = 0.1,
            full_labels: None = None,
            bounds: None = None,
            scores: None = None,
            recall_target: float = 0.95,
    ) -> LogEntry:
        """
        Use inclusion rate during random sample to extrapolate overall number of included records.
        Then stop when set recall target is reached based on estimate
        """
        n_incl_total = labels.sum()
        n_incl_sample = labels[~is_prioritised].sum()
        n_sample = (is_prioritised < 1).sum()

        prevalence = n_incl_sample / n_sample
        est_incl = int(prevalence * n_total)

        safe_to_stop = (est_incl > 0) and (n_incl_total / est_incl) > recall_target

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_incl=est_incl,
            score=(n_incl_total / est_incl),
            confidence_level=None,
        )
