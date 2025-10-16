

from typing import Generator, TypedDict

import numpy as np
import pandas as pd

from shared.method import Method, AbstractLogEntry, RECALL_TARGETS
from shared.types import IntList, FloatList

class HeuristicScoresParamSet(TypedDict):
    bound: str
    recall_target: float
    min_read_docs: int

class HeuristicScoresLogEntry(AbstractLogEntry):
    KEY: str = 'CHAO'
    bound: str
    recall_target: float
    min_read_docs: int
    est_recall: float


class Chao(Method):
    KEY: str = 'CHAO'

    def parameter_options(self) -> Generator[HeuristicScoresParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            yield HeuristicScoresParamSet(recall_target=recall_target, inclusion_threshold=0.5)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                recall_target: float = 0.95,
                inclusion_threshold: float = 0.5) -> HeuristicScoresLogEntry:
        """
        Using the estimate N_hat푁 and the corresponding 95% CI for N, we can determine if we can terminate
        the TAR procedure. The user can specify a recall target, such as 95% recall (note that the 95% recall
        target is not be confused with the 95% of the CI). The system tracks the estimate and CI to determine
        if the stopping criterion has been met. However, there are multiple ways to decide on the recall
        statistics and estimates.

        https://dl.acm.org/doi/10.1145/3724116

        via https://github.com/mpbron/allib-chao-experiments
        via https://github.com/mpbron/allib
        """
        y_pred = np.array(list_of_model_scores) >= inclusion_threshold
        y_true = np.array(list_of_labels)

        n_seen = len(y_true)
        n_incl = y_true.sum()

        if estimate > len(learner.env.dataset) or estimate == float("nan"):
            self.estimate = None
        else:
            self.estimate = estimate

        est_incl = y_pred.sum()
        seen_incl = y_true.sum()

        safe_to_stop = (est_incl > 0) and (seen_incl / est_incl) > recall_target

        return HeuristicScoresLogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_incl=est_incl,
            inclusion_threshold=inclusion_threshold,
        )
