from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

# TM QBCB
# https://dl.acm.org/doi/pdf/10.1145/3726302.3729879
# https://arxiv.org/pdf/2108.12746

class TargetParamSet(TypedDict):
    todo: float


class TargetLogEntry(AbstractLogEntry):
    KEY: str = 'TM-QBCB'
    todo: float


class Target(AbstractMethod):
    KEY: str = 'TM-QBCB'

    def parameter_options(self) -> Generator[TargetParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield TargetParamSet(todo=todo)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                todo: float = 1.0) -> TargetLogEntry:
        # TODO
        return TargetLogEntry(
            safe_to_stop=False,
            todo=todo,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(Target, TargetParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
