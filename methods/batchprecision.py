from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

# This is a stopping method based on tracking precision in the last batch_size records
# It stops when precision falls below a given threshold for the most recent batch

class BatchPrecisionParamSet(TypedDict):
    batch_size: int
    threshold: float

class BatchPrecisionLogEntry(AbstractLogEntry):
    KEY: str = 'BATCHPRECISION'
    batch_size: int
    threshold: float
    current_precision: float

class BatchPrecision(AbstractMethod):
    KEY: str = 'BATCHPRECISION'
    
    def parameter_options(self) -> Generator[BatchPrecisionParamSet, None, None]:
        for batch_size in [500, 1000, 2000]:
            for threshold in [0.05, 0.1, 0.2]:
                yield BatchPrecisionParamSet(batch_size=batch_size, threshold=threshold)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                batch_size: int = 1000,
                threshold: float = 0.1) -> BatchPrecisionLogEntry:
        
        # If we don't have enough data yet, it's not safe to stop
        if len(list_of_labels) < batch_size:
            return BatchPrecisionLogEntry(
                safe_to_stop=False,
                batch_size=batch_size,
                threshold=threshold,
                current_precision=1.0  # Assume perfect precision (i.e, definitely not safe to stop) if we don't have enough data
            )
        
        # Only consider the most recent batch_size elements
        last_batch_labels = np.array(list_of_labels[-batch_size:])
        
        # Calculate precision within this batch (% of relevant documents)
        last_batch_relevant_count = np.sum(last_batch_labels)
        current_precision = float(last_batch_relevant_count / batch_size)
        
        # Determine if it's safe to stop based on the precision threshold
        safe_to_stop = current_precision < threshold
        
        return BatchPrecisionLogEntry(
            safe_to_stop=safe_to_stop,
            batch_size=batch_size,
            threshold=threshold,
            current_precision=current_precision,
        )