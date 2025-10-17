from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    fraction: float


class LogEntry(_LogEntry, MethodParams):
    num_to_stop: int


class HeuristicFraction(Method[None, None, None, None]):
    KEY: str = 'HEURISTIC_FRAC'

    @classmethod
    def parameter_options(cls)  -> Generator[MethodParams, None, None]:
        for target in [.01, .05, .075, .1, 0.2]:
            yield MethodParams(fraction=target)

    @classmethod
    def compute(cls,
                n_total: int,
                labels: Labels,
                fraction: float = 0.05,
                is_prioritised: None = None,
                full_labels: None = None,
                scores: None = None,
                bounds: None = None,
                ) -> LogEntry:
        """
        Implements heuristic rule to stop after seeing N excluded records in a row.
        In this example, N = fraction * dataset size
        """
        num_to_stop = int(n_total * fraction)
        last_labels = labels[-min(len(labels), num_to_stop):]

        return LogEntry(KEY=cls.KEY,
                        safe_to_stop=1 not in last_labels,
                        num_to_stop=num_to_stop,
                        fraction=fraction,
                        confidence_level=None,
                        score=None,
                        recall_target=None)
