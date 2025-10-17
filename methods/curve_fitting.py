import warnings
import logging
from typing import Generator

import numpy as np

from scipy.stats import poisson
from scipy.optimize import curve_fit, OptimizeWarning

from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS, CONFIDENCE_TARGETS, NUM_WINDOWS
from shared.types import Labels


class MethodParams(_MethodParams):
    recall_target: float
    confidence_level: float
    n_windows: int


class LogEntry(_LogEntry, MethodParams):
    expected_includes: int | None
    curve_estimate: int | None


class CurveFitting(Method[None, None, None, None]):
    KEY: str = 'CURVE_FITTING'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for target in RECALL_TARGETS:
            for nw in NUM_WINDOWS:
                for conf in CONFIDENCE_TARGETS:
                    yield MethodParams(recall_target=target, confidence_level=conf, n_windows=nw)

    @classmethod
    def compute(
            cls,
            n_total: int,
            labels: Labels,
            recall_target: float = .07,
            confidence_level: float = 0.95,
            n_windows: int = 10,
            is_prioritised: None = None,
            full_labels: None = None,
            scores: None = None,
            bounds: None = None,
    ) -> LogEntry:
        """
        Implements stopping rule that fits a curve to the cumulative number of included records.
        Based on the convergence/limit of that curve, an expected number of relevant records is estimated

        Reference implementations
        # https://github.com/alisonsneyd/poisson_stopping_method/blob/master/run_stopping_point_algorithms.py
        # https://github.com/alisonsneyd/poisson_stopping_method/blob/master/utils/inhomogeneous_pp_fns.py

        :param recall_target: float = .07,  # 0.7  <- minimum desired recall level
        :param confidence_level: float = 0.95,  # 0.95  <- minimum desired probability for poisson
        :param n_windows: int = 10,  # 10  <- number of windows to make from sample
        """
        # min number rel docs must be initial sample to proceed with algorithm (orig: 20)
        min_rel_in_sample = 10

        # INHOMOGENEOUS POISSON PROCESS
        # check topic meets initial relevance requirement

        # if meet size requirement run algorithm; else return n_docs as stopping point
        if np.sum(labels) < min_rel_in_sample:
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=False,
                recall_target=recall_target,
                confidence_level=confidence_level,
                n_windows=n_windows,
                score=None,
                curve_estimate=None,
                expected_includes=None,
            )

        window_size = int(len(labels) / n_windows)
        windows = [(window_size * wi, window_size * (wi + 1))
                   for wi in range(n_windows - 1)]

        # x, y are points that will be used to fit curve
        # x-values are midpoints between start and end of windows
        x = np.array([round(np.mean([w_s, w_e]))
                      for (w_s, w_e) in windows])

        # y-values are the rate at which relevant documents occur in the window
        # ex: rate 0.1 = 0.1 rel docs per doc, or 1 in 10 docs are relevant
        y = np.array([np.sum(labels[w_s:w_e]) / window_size
                      for (w_s, w_e) in windows])

        # try to fit curve
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=OptimizeWarning)

                p0 = [0.1, 0.001, 1]  # initialise curve parameters
                opt, pcov = curve_fit(model_func, x, y, p0)  # fit curve
                a, k, b = opt

            # check distance between "curves" at end sample
            num_incl_seen = np.sum(labels)
            y2 = model_func(np.array(range(1, len(labels) + 1)), a, k, b)
            est_by_curve_end_samp = int(round(np.sum(y2)))

            num_incl_predicted = None
            if num_incl_seen >= recall_target * est_by_curve_end_samp:
                # using inhomogeneous Poisson process with fitted curve as rate fn, predict total number rel docs
                # integral model_func
                mu = (a / -k) * (np.exp(-k * n_total) - 1)
                # predict max number rel docs (using poisson cdf)
                num_incl_predicted = predict_n_rel(confidence_level, n_total, mu)

            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=(num_incl_predicted is not None and
                              num_incl_seen >= (recall_target * num_incl_predicted)),
                recall_target=recall_target,
                confidence_level=confidence_level,
                n_windows=n_windows,
                expected_includes=num_incl_predicted,
                curve_estimate=est_by_curve_end_samp,
                score=num_incl_seen / (recall_target * num_incl_predicted),
            )
        except Exception as e:
            # Failed to fit curve
            logging.exception(e)
            logging.warning('Failed to compute CurveFitting score, returning placeholder negative log')
            return LogEntry(
                KEY=cls.KEY,
                safe_to_stop=False,
                recall_target=recall_target,
                confidence_level=confidence_level,
                n_windows=n_windows,
                score=None,
                curve_estimate=None,
                expected_includes=None,
            )


# fn to fit curve to points
def model_func(x, a, k, b):  # x = vector x values
    return a * np.exp(-k * x)


# function to predict max number of relevant documents
def predict_n_rel(confidence_level: float, n_docs: int, mu: float):
    i = 0
    cum_prob = poisson.cdf(i, mu)
    while (i < n_docs) and (cum_prob < confidence_level):
        i += 1
        cum_prob = poisson.cdf(i, mu)

    return i
