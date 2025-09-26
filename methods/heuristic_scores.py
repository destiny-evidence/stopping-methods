from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS, INCLUSION_THRESHOLDS
from shared.types import Labels, Scores
from typing import Generator


class MethodParams(_MethodParams):
    recall_target: float
    inclusion_threshold: float


class LogEntry(_LogEntry, MethodParams):
    est_incl: int


class HeuristicScores(Method[Scores, None, None, None]):
    KEY: str = 'HEURISTIC_SCORES'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for recall_target in RECALL_TARGETS:
            for it in INCLUSION_THRESHOLDS:
                yield MethodParams(recall_target=recall_target, inclusion_threshold=it)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            scores: Scores,
            batch_size: int = 1000,
            threshold: float = 0.1,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
            recall_target: float = 0.95,
            inclusion_threshold: float = 0.5,
    ) -> LogEntry:
        """
        Use model scores to estimate the number of included documents and then
        stop when the number of seen includes is above the recall target

        Inspired by https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/heuristic.py#L80
        """
        y_true = labels >= 0
        y_pred = scores >= inclusion_threshold
        est_incl = y_pred.sum()
        seen_incl = y_true.sum()

        safe_to_stop = (est_incl > 0) and (seen_incl / est_incl) > recall_target

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_incl=est_incl,
            inclusion_threshold=inclusion_threshold,
            score=(seen_incl / est_incl),
            confidence_level=None,
        )
