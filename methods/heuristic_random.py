from typing import Generator, TypedDict

import numpy as np
import pandas as pd

from shared.method import AbstractMethod, AbstractLogEntry, RECALL_TARGETS
from shared.types import IntList, FloatList


class HeuristicRandomParamSet(TypedDict):
    recall_target: float


class HeuristicRandomLogEntry(AbstractLogEntry):
    KEY: str = 'HEURISTIC_RANDOM'
    recall_target: float
    est_incl: int


class HeuristicRandom(AbstractMethod):
    KEY: str = 'HEURISTIC_RANDOM'

    def parameter_options(self) -> Generator[HeuristicRandomParamSet, None, None]:
        for recall_target in RECALL_TARGETS:
            yield HeuristicRandomParamSet(recall_target=recall_target)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                recall_target: float = 0.95) -> HeuristicRandomLogEntry:
        """
        Use inclusion rate during random sample to extrapolate overall number of included records.
        Then stop when set recall target is reached based on estimate
        """
        # human annotations
        y_true = np.array(list_of_labels)
        seen_incl = y_true.sum()

        # mask and number of labels that are randomly screened
        random_sample = np.array(is_prioritised) == 0
        n_random = random_sample.sum()

        # compute inclusion rate from random sample and extrapolate
        incl_rate = y_true[random_sample].sum() / n_random
        est_incl = int(incl_rate * len(list_of_model_scores))

        safe_to_stop = (est_incl > 0) and (seen_incl / est_incl) > recall_target

        return HeuristicRandomLogEntry(
            safe_to_stop=safe_to_stop,
            recall_target=recall_target,
            est_incl=est_incl,
        )
