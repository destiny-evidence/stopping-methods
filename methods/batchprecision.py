import numpy as np
from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    batch_size: int
    threshold: float


class LogEntry(_LogEntry, MethodParams):
    current_precision: float


class BatchPrecision(Method[None, None, None, None]):
    KEY: str = 'BATCHPRECISION'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for batch_size in [500, 1000, 2000]:
            for threshold in [0.05, 0.1, 0.2]:
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
        This is a stopping method based on tracking precision in the last batch_size records
        It stops when precision falls below a given threshold for the most recent batch
        """

        # If we don't have enough data yet, it's not safe to stop
        if len(labels) < batch_size:
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=False,
                batch_size=batch_size,
                threshold=threshold,
                # Assume perfect precision (i.e, definitely not safe to stop) if we don't have enough data
                current_precision=1.0,
                score=1.0,
                confidence_level=None,
                recall_target=None,
            )

        # Only consider the most recent batch_size elements
        last_batch_labels = labels[-batch_size:]

        # Calculate precision within this batch (% of relevant documents)
        last_batch_relevant_count = np.sum(last_batch_labels)
        current_precision = float(last_batch_relevant_count / batch_size)

        return LogEntry(
            KEY=cls.KEY,
            # Determine if it's safe to stop based on the precision threshold
            safe_to_stop=current_precision < threshold,
            batch_size=batch_size,
            threshold=threshold,
            current_precision=current_precision,
            confidence_level=None,
            recall_target=None,
            score=current_precision,
        )
