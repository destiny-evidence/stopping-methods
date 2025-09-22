from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList


# https://github.com/ReemBinHezam/RLStop
# via https://arxiv.org/abs/2405.02525

class RLStopParamSet(TypedDict):
    todo: float


class RLStopLogEntry(AbstractLogEntry):
    KEY: str = 'RLStop'
    todo: float


class RLStop(AbstractMethod):
    KEY: str = 'RLStop'

    def parameter_options(self) -> Generator[RLStopParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield RLStopParamSet(todo=todo)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                todo: float = 1.0) -> RLStopLogEntry:
        # TODO
        return RLStopLogEntry(
            safe_to_stop=False,
            todo=todo,
        )
if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(RLStop, RLStopParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
