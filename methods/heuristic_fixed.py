from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    num_to_stop: int


class LogEntry(_LogEntry, MethodParams):
    pass


class HeuristicFixed(Method[None, None, None, None]):
    KEY: str = 'HEURISTIC_FIX'

    def parameter_options(self) -> Generator[MethodParams, None, None]:
        for target in [50, 100, 200, 300]:
            yield MethodParams(num_to_stop=target)

    @classmethod
    def compute(cls,
                n_total: int,
                labels: Labels,
                num_to_stop: int = 20,
                is_prioritised: None = None,
                full_labels: None = None,
                scores: None = None,
                bounds: None = None,
                ) -> LogEntry:
        """
        Implements heuristic rule to stop after seeing N excluded records in a row.
        """
        last_labels = labels[-min(len(labels), num_to_stop):]

        return LogEntry(KEY=cls.KEY,
                        safe_to_stop=1 not in last_labels,
                        num_to_stop=num_to_stop,
                        confidence_level=None,
                        score=None,
                        recall_target=None)
