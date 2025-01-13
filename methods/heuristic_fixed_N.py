from typing import Generator, TypedDict

import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

Array = np.ndarray[tuple[int], np.dtype[np.int64]]


class HeuristicFixedNParamset(TypedDict):
    stoppingN: int


class HeuristicFixedNLogEntry(AbstractLogEntry):
    KEY: str = 'HEURISTICFIXEDN'
    num_to_stop: int


class HeuristicFixedN(AbstractMethod):
    KEY: str = 'HEURISTICFIXEDN'

    def parameter_options(self) -> Generator[HeuristicFixedNParamset, None, None]:
        for target in [50, 100, 200, 300]:
            yield HeuristicFixedNParamset(stoppingN=target)

    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                stoppingN: int) -> HeuristicFixedNLogEntry:

        try:#When testing updated code with max function (now in except statement) I realized there might be an issue because it seems to just take all screened refs and thus doesn't stop (as there usually is a 1 somewhre in early screening part). I know it was added because there was an edge-case somewhere which led the original code using [-num_to_stop:] to fail so I added some safety statements and the new code as fallback.
            last_labels = list(list_of_labels)[-stoppingN:]
        except:
            last_labels = list_of_labels[max(0, -stoppingN):]

        return HeuristicFixedNLogEntry(safe_to_stop=1 not in last_labels,
                                         num_to_stop=stoppingN)
