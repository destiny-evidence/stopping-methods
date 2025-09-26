from typing import Annotated, Literal

import pandas as pd
import numpy as np
import numpy.typing as npt

BatchBounds = Annotated[npt.NDArray[np.int64], Literal['N', 3]]
Bounds = Annotated[npt.NDArray[np.int64], Literal['N', 2]]
Scores = Annotated[npt.NDArray[np.float64], Literal['N']]
BinaryScores = Annotated[npt.NDArray[np.float64], Literal['N', 2]]
Labels = Annotated[npt.NDArray[np.int64], Literal['N']]
Sampling = Annotated[npt.NDArray[np.bool], Literal['N']]

type StrList = list[str] | pd.Series
type IntList = list[int] | Labels | pd.Series
type FloatList = list[float] | Scores | pd.Series
type Indices = list[int] | Labels | pd.Series
type Mask = list[bool] | Sampling | pd.Series

__all__ = ['IntList', 'FloatList', 'StrList', 'Mask', 'Indices',
           'BatchBounds', 'Bounds', 'Scores', 'Labels', 'Sampling', 'BinaryScores']
