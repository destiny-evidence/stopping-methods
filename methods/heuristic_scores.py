from typing import Generator, TypedDict

import numpy as np
import pandas as pd

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS
from shared.types import IntList, FloatList

class HeuristicScoresParamSet(TypedDict):
    recall_target: float
    inclusion_threshold: float


class HeuristicScoresLogEntry(AbstractLogEntry):
    KEY: str = 'HEURISTIC_SCORES'
    recall_target: float
    inclusion_threshold: float
    est_incl: int


class HeuristicScores(AbstractMethod):
    KEY: str = 'HEURISTIC_SCORES'

    def parameter_options(self) -> Generator[HeuristicScoresParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            yield HeuristicScoresParamSet(recall_target=recall_target, inclusion_threshold=0.5)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                recall_target: float = 0.95,
                inclusion_threshold: float = 0.5) -> HeuristicScoresLogEntry:
        """
        Use model scores to estimate the number of included documents and then
        stop when the number of seen includes is above the recall target

        Inspired by https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/heuristic.py#L80
        """
        y_pred = np.array(list_of_model_scores) >= inclusion_threshold
        y_true = np.array(list_of_labels) >= inclusion_threshold
        est_incl = y_pred.sum()
        seen_incl = y_true.sum()

        safe_to_stop = (est_incl > 0) and (seen_incl / est_incl) > recall_target

        return HeuristicScoresLogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_incl=est_incl,
            inclusion_threshold=inclusion_threshold,
        )
