from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList


# https://github.com/elevatelaw/ICAIL2023_confidence_sequences
# via https://dl.acm.org/doi/abs/10.1145/3594536.3595167

class ConfSeqParamSet(TypedDict):
    todo: float


class ConfSeqLogEntry(AbstractLogEntry):
    KEY: str = 'CONF'
    todo: float


class ConfSeq(AbstractMethod):
    KEY: str = 'CONF'

    def parameter_options(self) -> Generator[ConfSeqParamSet, None, None]:
        for todo in [1.0, 1.1, 1.2]:
            yield ConfSeqParamSet(todo=todo)

    @classmethod
    def compute(cls,
                dataset_size: int,
                list_of_labels: IntList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray | None = None,
                list_of_model_scores: FloatList | None = None,
                todo: float = 1.0) -> ConfSeqLogEntry:
        # TODO
        return ConfSeqLogEntry(
            safe_to_stop=False,
            todo=todo,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    dataset, results = test_method(ConfSeq, ConfSeqParamSet(todo=1.0), 2)
    fig, ax = plots(dataset, results)
    fig.show()
