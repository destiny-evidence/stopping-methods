from numpy.typing import NDArray
import pandas as pd

type StrList = list[str] | pd.Series
type IntList = list[int] | NDArray[int] | pd.Series
type FloatList = list[float] | NDArray[float] | pd.Series
type Indices = list[int] | NDArray[int] | pd.Series
type Mask =  list[bool] | NDArray[bool] | pd.Series

__all__ = ['IntList', 'FloatList', 'StrList', 'Mask', 'Indices']
