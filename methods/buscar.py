from typing import Generator

import numpy as np
import numpy.typing as npt

from scipy.stats import hypergeom, nchypergeom_wallenius

from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS, CONFIDENCE_TARGETS
from shared.types import Labels, IntList


class MethodParams(_MethodParams):
    bias: float
    recall_target: float
    confidence_level: float


class LogEntry(_LogEntry, MethodParams):
    pass


class Buscar(Method[None, None, None, None]):
    KEY: str = 'BUSCAR'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for tr in RECALL_TARGETS:
            for ci in CONFIDENCE_TARGETS:
                for bias in [1., 2., 5., 10.]:
                    yield MethodParams(recall_target=tr, bias=bias, confidence_level=ci)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            confidence_level: float = 0.95,
            recall_target: float = 0.95,
            bias: float = 1.0,
            scores: None = None,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        score = calculate_h0(
            labels_=labels,
            n_docs=n_total,
            recall_target=recall_target,
            bias=bias,
        )

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=score is not None and score < 1 - confidence_level,
            score=score,
            confidence_level=confidence_level,
            recall_target=recall_target,
            bias=bias
        )


Array = np.ndarray[tuple[int], np.dtype[np.int64]]


def calculate_h0(labels_: IntList, n_docs: int, recall_target: float = .95, bias: float = 1.) -> float | None:
    """
    Calculates a p-score for our null hypothesis h0, that we have missed our recall target `recall_target`.

    :param labels_: An ordered sequence of 1s and 0s representing, in the order
        in which they were screened, relevant and irrelevant documents
        respectively.
    :param n_docs: The total number of documents from which you want to find the
        relevant examples. The size of the haystack.
    :param recall_target: The proportion of truly relevant documents you want
        to find, defaults to 0.95
    :param bias: The assumed likelihood of drawing a random relevant document
        over the likelihood of drawing a random irrelevant document. The higher
        this is, the better our ML has worked. When this is different to 1,
        we calculate the p score using biased urns.
    :return: p-score for our null hypothesis.
             We can reject the null hypothesis (and stop screening) if p is below 1 - our confidence level.

    """
    labels: Array = (labels_ if type(labels_) is np.ndarray
                     else np.array(labels_, dtype=np.int_))

    # Number of relevant documents we have seen
    r_seen = labels.sum()

    # Reverse the list so we can later construct the urns
    urns = labels[::-1]  # Urns of previous 1,2,...,N documents
    urn_sizes = np.arange(urns.shape[0]) + 1  # The sizes of these urns

    # Now we calculate k_hat, which is the minimum number of documents there would have to be
    # in each of our urns for the urn to be in keeping with our null hypothesis
    # that we have missed our target
    k_hat = np.floor(
        r_seen / recall_target + 1 -  # Divide num of relevant documents by our recall target and add 1  # noqa: W504
        (
                r_seen -  # from this we subtract the total relevant documents seen  # noqa: W504
                urns.cumsum()  # before each urn
        )
    )

    # Test the null hypothesis that a given recall target has been missed
    p: npt.NDArray[np.float64]
    if bias == 1:
        p = hypergeom.cdf(  # the probability of observing
            urns.cumsum(),  # the number of relevant documents in the sample
            n_docs - (urns.shape[0] - urn_sizes),  # In a population made up out of the urn and all remaining docs
            k_hat,  # Where K_hat docs in the population are actually relevant
            urn_sizes  # After observing this many documents
        )
    else:
        p = nchypergeom_wallenius.cdf(
            urns.cumsum(),  # the number of relevant documents in the sample
            n_docs - (urns.shape[0] - urn_sizes),  # In a population made up out of the urn and all remaining docs
            k_hat,  # Where K_hat docs in the population are actually relevant
            urn_sizes,  # After observing this many documents
            bias  # Where we are bias times more likely to pick a random relevant document
        )

    # We computed this for all, so only return the smallest
    p_min: float = p.min()

    if np.isnan(p_min):
        return None
    return p_min


if __name__ == '__main__':
    from shared.test import test_method, plots

    p = MethodParams(recall_target=0.95, bias=1.0, confidence_level=0.99)
    dataset, results = test_method(Buscar, p, 2)
    fig, _ = plots(dataset, results, p)
    fig.show()

    p = MethodParams(recall_target=0.9, bias=1.0, confidence_level=0.99)
    dataset, results = test_method(Buscar, p, 2)
    fig, _ = plots(dataset, results, p)
    fig.show()

    p = MethodParams(recall_target=0.95, bias=5.0, confidence_level=0.99)
    dataset, results = test_method(Buscar, p, 2)
    fig, _ = plots(dataset, results, p)
    fig.show()

    p = MethodParams(recall_target=0.9, bias=5.0, confidence_level=0.99)
    dataset, results = test_method(Buscar, p, 2)
    fig, _ = plots(dataset, results, p)
    fig.show()
