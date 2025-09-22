from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS, INCLUSION_THRESHOLDS
from shared.types import IntList, FloatList


class AprioriParamSet(TypedDict):
    recall_target: float
    inclusion_threshold: float


class AprioriLogEntry(AbstractLogEntry):
    KEY: str = 'APRIORI'
    recall_target: float
    inclusion_threshold: float
    est_recall: float


class Apriori(AbstractMethod):
    KEY: str = 'APRIORI'

    def parameter_options(self) -> Generator[AprioriParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            for inclusion_threshold in INCLUSION_THRESHOLDS:
                yield AprioriParamSet(recall_target=recall_target, inclusion_threshold=inclusion_threshold)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                recall_target: float = 0.95,
                inclusion_threshold: float = 0.5) -> AprioriLogEntry:
        # inspired by https://github.com/mpbron/allib/blob/stable/allib/stopcriterion/apriori.py#L18
        y_pred = np.array(list_of_model_scores[:len(list_of_labels)]) >= inclusion_threshold
        y_true = np.array(list_of_labels) == 1

        recall = recall_score(y_true, y_pred, zero_division=0)

        safe_to_stop = (0 < recall < 1) and recall > recall_target

        return AprioriLogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_recall=recall,
            inclusion_threshold=inclusion_threshold,
        )
