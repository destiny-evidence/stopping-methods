import numpy as np
from scipy.stats import betabinom

from shared.method import Method, _MethodParams, _LogEntry
from shared.types import Labels
from typing import Generator


class MethodParams(_MethodParams):
    pass


class LogEntry(_LogEntry, MethodParams):
    pass


class ConfSeq(Method[None, None, None, None]):
    KEY: str = 'CONF'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for batch_size in [500, 1000, 2000]:
            yield MethodParams(batch_size=batch_size, threshold=threshold)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            scores: None = None,
            is_prioritised: None = None,
            full_labels: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements confidence sequence rule
        > Lewis et al., ICAIL'23. "Confidence Sequences for Evaluating One-Phase Technology-Assisted Review"
        > via https://dl.acm.org/doi/abs/10.1145/3594536.3595167

        Reference implementation:
        https://github.com/elevatelaw/ICAIL2023_confidence_sequences
        -> implementation broken!
           see https://github.com/elevatelaw/ICAIL2023_confidence_sequences/issues

        """
        n_seen = len(labels)  # M
        n_seen_incl = sum(labels)#M+
        n_seen_excl = n_seen - n_seen_incl#M-

        prior_a = 1/99
        prior_b = 1
        prior_dist = betabinom(n_seen, prior_a, prior_b)

        n_unseen = n_total-n_seen
        posterior_a = prior_a + n_seen_incl
        posterior_b = prior_b + n_seen_excl
        posterior_dist = betabinom(n_unseen, posterior_a, posterior_b)

        est_incl_min = n_seen_incl#Nplus_min
        est_incl_max = n_total - n_seen_excl#Nplus_max
        est_incl_range = np.arange(est_incl_min, est_incl_max + 1)

        priors = np.array([prior_dist.pmf(n) for n in est_incl_range])
        posteriors = np.array([prior_dist.pmf(n) for n in est_incl_range])
        confidence_intervals = est_incl_range[priors < (20*posteriors)]

        est_min = confidence_intervals.min()
        est_max = confidence_intervals.max()

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
    dataset, results = test_method(ConfSeq, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
