from typing import Generator, Type

from shared.dataset import RankedDataset
from shared.method import AbstractMethod

from .alison import Alison
from .apriori import Apriori
from .batchprecision import BatchPrecision
from .buscar import Buscar
from .curve_fitting import CurveFitting
from .heuristic_fixed import HeuristicFixed
from .heuristic_fractions import HeuristicFraction
from .heuristic_scores import HeuristicScores
from .heuristic_random import HeuristicRandom
from .knee import Knee
from .method2399 import Method2399
from .quant_ci import QuantCI

# from .conf import ConfSeq
# from .scal import SCAL
# from .rlstop import RLStop
# from .chao import Chao

__all__ = [
    'Alison',
    'Apriori',
    'BatchPrecision',
    'Buscar',
    'CurveFitting',
    'HeuristicFixed',
    'HeuristicFraction',
    'HeuristicScores',
    'HeuristicRandom',
    'Knee',
    'Method2399',
    'QuantCI',
    # 'ConfSeq',
    # 'SCAL',
    # 'RLStop',
    # 'Chao'
]


def get_methods(methods: list[str] | None = None) -> Generator[Type[AbstractMethod], None, None]:
    methods_incl = set(methods or [])
    for Method in map(globals().get, __all__):
        if Method.KEY in methods_incl or methods is None:
            yield Method


def it_methods(dataset: RankedDataset, methods: list[str] | None = None) -> Generator[AbstractMethod, None, None]:
    for Method in get_methods(methods=methods):
        yield Method(dataset)
