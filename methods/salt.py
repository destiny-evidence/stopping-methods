from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList


# https://github.com/levnikmyskin/salt
# https://link.springer.com/article/10.1007/s10618-023-00961-5

class SALTParamSet(TypedDict):
    todo: float


class SALTLogEntry(AbstractLogEntry):
    KEY: str = 'SALτ'
    todo: float


class SALT(AbstractMethod):
    KEY: str = 'SALτ'

    def parameter_options(self) -> Generator[SALTParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield SALTParamSet(todo=todo)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                todo: float = 1.0) -> SALTLogEntry:
        # TODO
        return SALTLogEntry(
            safe_to_stop=False,
            todo=todo,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(SALT, SALTParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
