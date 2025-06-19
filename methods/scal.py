from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList


# https://dl.acm.org/doi/abs/10.1145/2983323.2983776
# via https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/scal.py

class SCALParamSet(TypedDict):
    todo: float


class SCALLogEntry(AbstractLogEntry):
    KEY: str = 'S-CAL'
    todo: float


class SCAL(AbstractMethod):
    KEY: str = 'S-CAL'

    def parameter_options(self) -> Generator[SCALParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield SCALParamSet(todo=todo)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                todo: float = 1.0) -> SCALLogEntry:
        # TODO
        return SCALLogEntry(
            safe_to_stop=False,
            todo=todo,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(SCAL, SCALParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
