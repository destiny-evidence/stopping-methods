import numpy as np
import pandas as pd

type IntList = np.ndarray[tuple[int], np.dtype[np.int_]] | list[int] | pd.Series[int]
type FloatList = np.ndarray[tuple[float], np.dtype[np.int_]] | list[float] | pd.Series[float]
type StrList = list[str] | pd.Series[str]


__all__ = ['IntList', 'FloatList', 'StrList']
