from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

# This is a stopping method based on the formula: (from Cormack and Grossman, 2015)
# n_reviewed ≥ alpha * num_relevant_reviewed + 2399
# Where alpha can be 1, 1.1, or 1.2

class Method2399ParamSet(TypedDict):
    alpha: float
    constant: int

class Method2399LogEntry(AbstractLogEntry):
    KEY: str = 'METHOD2399'
    alpha: float
    constant: int
    num_reviewed: int
    num_relevant_reviewed: int
    threshold: float

class Method2399(AbstractMethod):
    KEY: str = 'METHOD2399'
    
    def parameter_options(self) -> Generator[Method2399ParamSet, None, None]:
        for alpha in [1.0, 1.1, 1.2]:
            yield Method2399ParamSet(alpha=alpha, constant=2399)
    
    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                alpha: float = 1.0,
                constant: int = 2399) -> Method2399LogEntry:
        
        # Calculate the number of documents reviewed so far
        num_reviewed = len(list_of_labels)
        
        # Calculate the number of relevant documents found so far
        num_relevant_reviewed = int(np.sum(list_of_labels))
        
        # Calculate the threshold based on the formula
        threshold = alpha * num_relevant_reviewed + constant
        
        # Determine if it's safe to stop based on the heuristic formula
        safe_to_stop = num_reviewed >= threshold
        
        return Method2399LogEntry(
            safe_to_stop=safe_to_stop,
            alpha=alpha,
            constant=constant,
            num_reviewed=num_reviewed,
            num_relevant_reviewed=num_relevant_reviewed,
            threshold=threshold,
        )