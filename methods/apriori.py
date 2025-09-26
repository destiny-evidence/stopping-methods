from sklearn.metrics import recall_score

from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS, INCLUSION_THRESHOLDS
from shared.types import Labels, Scores
from typing import Generator


class MethodParams(_MethodParams):
    recall_target: float
    inclusion_threshold: float


class LogEntry(_LogEntry, MethodParams):
    est_recall: float


class Apriori(Method[Scores, None, None, None]):
    KEY: str = 'APRIORI'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for recall_target in RECALL_TARGETS:
            for inclusion_threshold in INCLUSION_THRESHOLDS:
                yield MethodParams(recall_target=recall_target, inclusion_threshold=inclusion_threshold)

    @classmethod
    def compute(
            cls,
            n_total: int,  # Total number of records in datasets (seen + unseen)
            labels: Labels,
            scores: Scores,
            recall_target: float = 0.95,
            inclusion_threshold: float = 0.5,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:

        # inspired by https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/apriori.py#L18
        y_pred = scores[:len(labels)] >= inclusion_threshold
        y_true = labels == 1

        recall = recall_score(y_true, y_pred, zero_division=0)

        safe_to_stop = (0 < recall < 1) and recall > recall_target

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_recall=recall,
            inclusion_threshold=inclusion_threshold,
            confidence_level=None,
            score=None,
        )
