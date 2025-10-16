from typing import Generator, Type

from shared.method import Method

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
from .tm_qbcb import TargetQBCB
from .ipp import PointProcess
from .salt import SALT
from .scal import SCAL

__all__ = [
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
    'TargetQBCB',
    'PointProcess',
    'SALT',
    'SCAL',
]


def it_methods(methods: list[str] | None = None) -> Generator[Type[Method], None, None]:
    methods_incl = set(methods or [])
    for Method in map(globals().get, __all__):
        if Method.KEY in methods_incl or methods is None:
            yield Method
