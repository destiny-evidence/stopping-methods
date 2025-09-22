

from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList


# https://github.com/ReemBinHezam/TAR_Stopping_Point_Processes
# via https://dl.acm.org/doi/abs/10.1145/3631990

class PointProcessParamSet(TypedDict):
    todo: float


class PointProcessLogEntry(AbstractLogEntry):
    KEY: str = 'PPO'
    todo: float


class PointProcess(AbstractMethod):
    KEY: str = 'PPO'

    def parameter_options(self) -> Generator[PointProcessParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield PointProcessParamSet(todo=todo)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                todo: float = 1.0) -> PointProcessLogEntry:
        # TODO
        return PointProcessLogEntry(
            safe_to_stop=False,
            todo=todo,
        )
if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(PointProcess, PointProcessParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
