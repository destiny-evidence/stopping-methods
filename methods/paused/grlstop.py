from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    pass


class LogEntry(_LogEntry, MethodParams):
    pass


class GRLStop(Method[None, None, None, None]):
    KEY: str = 'GRLStop'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for batch_size in [500, 1000, 2000]:
            yield MethodParams(batch_size=batch_size, threshold=threshold)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            batch_size: int = 1000,
            threshold: float = 0.1,
            scores: None = None,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements GRLStop

        > Bin-Hezam and Stevenson, SIGIR 2025. "A generalised and adaptable reinforcement learning stopping method"
        > via https://doi.org/10.1145/3726302.3729879

        Reference implementation:
        https://github.com/ReemBinHezam/GRLStop
        """

        # TODO
        raise NotImplemented
        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=False,
            score=1.0,
            confidence_level=None,
            recall_target=None,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = MethodParams()
    dataset, results = test_method(GRLStop, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
