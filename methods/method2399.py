import numpy as np

from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    alpha: float
    constant: int


class LogEntry(_LogEntry, MethodParams):
    num_reviewed: int
    num_relevant_reviewed: int
    threshold: float


class Method2399(Method[None, None, None, None]):
    KEY: str = 'METHOD2399'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for alpha in [1.0, 1.1, 1.2]:
            yield MethodParams(alpha=alpha, constant=2399)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            alpha: float = 1.0,
            constant: int = 2399,
            scores: None = None,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        This is a stopping method based on the formula
        n_reviewed ≥ alpha * num_relevant_reviewed + 2399
        Where alpha can be 1, 1.1, or 1.2

        Published by Cormack and Grossman, 2015
        """
        # Calculate the number of documents reviewed so far
        num_reviewed = len(labels)

        # Calculate the number of relevant documents found so far
        num_relevant_reviewed = int(np.sum(labels))

        # Calculate the threshold based on the formula
        threshold = alpha * num_relevant_reviewed + constant

        # Determine if it's safe to stop based on the heuristic formula
        safe_to_stop = num_reviewed >= threshold

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            alpha=alpha,
            constant=constant,
            num_reviewed=num_reviewed,
            num_relevant_reviewed=num_relevant_reviewed,
            threshold=threshold,
            score=None,
        )
