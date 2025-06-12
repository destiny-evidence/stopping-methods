from typing import Generator, Type

from shared.dataset import RankedDataset
from shared.method import AbstractMethod

from .buscar import Buscar
from .heuristic_fractions import HeuristicFraction
from .heuristic_fixed import HeuristicFixed
from .curve_fitting import CurveFitting
from .knee import Knee
from .alison import Alison
from .batchprecision import BatchPrecision
from .method2399 import Method2399
from .apriori import Apriori

__all__ = [
    'Buscar',
    'Knee',
    'CurveFitting',
    'HeuristicFraction',
    'HeuristicFixed',
    'Alison',
    'BatchPrecision',
    'Method2399',
    'Apriori',
]


def get_methods(methods: list[str] | None = None) -> Generator[Type[AbstractMethod], None, None]:
    methods_incl = set(methods or [])
    for Method in map(globals().get, __all__):
        if Method.KEY in methods_incl or methods is None:
            yield Method


def it_methods(dataset: RankedDataset, methods: list[str] | None = None) -> Generator[AbstractMethod, None, None]:
    for Method in get_methods(methods=methods):
        yield Method(dataset)
