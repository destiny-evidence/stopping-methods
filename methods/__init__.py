from typing import Generator

from shared.dataset import RankedDataset
from shared.method import AbstractMethod
from .buscar import Buscar
from .heuristic_fractions import HeuristicFraction
from .heuristic_fixed import HeuristicFixed
from .curve_fitting import CurveFitting

__all__ = ['Buscar', 'CurveFitting', 'HeuristicFraction', 'HeuristicFixed']


def it_methods(dataset: RankedDataset) -> Generator[AbstractMethod, None, None]:
    for Method in map(globals().get, __all__):
        yield Method(dataset)
