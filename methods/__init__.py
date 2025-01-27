from typing import Generator

from shared.dataset import RankedDataset
from shared.method import AbstractMethod

from .buscar import Buscar
from .heuristic_fractions import HeuristicFraction
from .heuristic_fixed import HeuristicFixed
from .curve_fitting import CurveFitting
from .knee import Knee
from .alison import Alison

__all__ = ['Buscar', 'CurveFitting', 'HeuristicFraction', 'HeuristicFixed', 'Alison', 'Knee']


def it_methods(dataset: RankedDataset, methods: list[str] | None = None) -> Generator[AbstractMethod, None, None]:
    methods_incl = set(methods or [])
    for Method in map(globals().get, __all__):
        if Method.KEY in methods_incl or methods is None:
            yield Method(dataset)
