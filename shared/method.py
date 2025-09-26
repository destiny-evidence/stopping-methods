from abc import ABC, abstractmethod
from typing import Generator, TypedDict

from shared.types import Bounds, Labels, Scores, Sampling
from typing import Generic, TypeVar

T_scores = TypeVar("T_scores", bound=Scores | None)
T_labels = TypeVar("T_labels", bound=Labels | None)
T_sampling = TypeVar("T_sampling", bound=Sampling | None)
T_bounds = TypeVar("T_bounds", bound=Bounds | None)


RECALL_TARGETS = [.8, .9, .95, .99]
CONFIDENCE_TARGETS = [.8, .9, .95, .99]
INCLUSION_THRESHOLDS = [.25, .5, .75, .9]
WINDOW_SIZES = [50, 500, 1000]
NUM_WINDOWS = [5, 10, 20, 50, 100]


class _MethodParams(TypedDict, total=False):
    recall_target: float | None
    confidence_level: float | None


class _LogEntry(TypedDict):
    KEY: str
    safe_to_stop: bool
    score: float | None


class Method(ABC, Generic[T_scores, T_labels, T_bounds, T_sampling]):
    KEY: str

    @classmethod
    @abstractmethod
    def parameter_options(cls) -> Generator[_MethodParams, None, None]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def compute(cls,
                n_total: int,  # Total number of records in datasets (seen + unseen)
                labels: Labels,  # Seen labels
                scores: T_scores,  # Model ranking scores for entire set (optional for some methods)
                is_prioritised: T_sampling,  # Mask for randomly sampled data (optional for some methods)
                full_labels: T_labels,  # Full annotation, only for fully annotated datasets to simulate target methods
                bounds: T_bounds,  # All batch boundaries in `labels` (optional for some methods)
                **kwargs: _MethodParams) -> _LogEntry:
        raise NotImplementedError()

    @classmethod
    def retrospective(cls,
                      n_total: int,
                      labels: Labels,
                      scores: T_scores,
                      is_prioritised: T_sampling,
                      bounds: T_bounds,
                      batch_size: int = 100,
                      **kwargs: _MethodParams) -> Generator[_LogEntry, None, None]:
        for n_seen_batch in range(batch_size, len(labels), batch_size):
            yield cls.compute(n_total=n_total,
                              labels=labels[:n_seen_batch],
                              scores=scores,
                              is_prioritised=is_prioritised[:n_seen_batch],
                              bounds=bounds[:n_seen_batch],
                              full_labels=labels,
                              **kwargs)
