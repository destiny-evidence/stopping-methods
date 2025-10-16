from enum import Enum
from typing import Generator

import numpy as np

from shared.types import Labels, Scores
from shared.method import Method, _MethodParams, _LogEntry, RECALL_TARGETS


class StoppingCondition(str, Enum):
    loose = 'LOOSE'
    strict_v1 = 'strict_v2'
    strict_v2 = 'strict_v2'


class MethodParams(_MethodParams):
    stopping_condition: StoppingCondition
    recall_target: float


class LogEntry(_LogEntry, MethodParams):
    est_incl: int


class AutoStop(Method[Scores, None, None, None]):
    KEY: str = 'AutoStop'

    @classmethod
    def parameter_options(cls) -> Generator[MethodParams, None, None]:
        for tr in RECALL_TARGETS:
            for sc in [StoppingCondition.loose, StoppingCondition.strict_v1, StoppingCondition.strict_v2]:
                yield MethodParams(stopping_condition=sc, recall_target=tr)

    @classmethod
    def compute(
            cls,
            *args,
            n_total: int,
            labels: Labels,
            scores: Scores,
            recall_target: float = 0.95,
            stopping_condition: StoppingCondition = StoppingCondition.loose,
            **kwargs,
    ) -> LogEntry:
        """
        Reference implementation:
        https://github.com/dli1/auto-stop-tar/blob/master/autostop/tar_model/auto_stop.py
        """

        entry = LogEntry(
            KEY=cls.KEY,
            safe_to_stop=False,
            score=None,
            recall_target=recall_target,
            stopping_condition=stopping_condition,
            confidence_level=None,
        )

        n_seen = len(labels)
        scores_seen = np.nan_to_num(scores[:n_seen], copy=True, nan=0.99)

        # first order inclusion probabilities
        probs_1o = n_seen * np.log(1 - scores_seen)
        probs_1o = 1.0 - np.exp(probs_1o)
        total_1o = np.sum(labels / probs_1o)

        if np.min(probs_1o) <= 0 or np.max(probs_1o) > 1.0:
            return entry

        var_1 = -1
        var_2 = -1
        if stopping_condition is StoppingCondition.loose:
            pass
        else:
            # second order inclusion probabilities
            temp = np.tile(scores_seen, (n_seen, 1))
            temp = 1.0 - temp - temp.T
            np.fill_diagonal(temp, 1)
            probs_2o = n_total * np.log(temp)
            temp = np.tile(probs_1o, (n_seen, 1))
            probs_2o = temp + temp.T - (1 - np.exp(probs_2o))

            if np.min(probs_2o) <= 0:
                return entry

            if stopping_condition is StoppingCondition.strict_v1:
                temp = np.tile(probs_1o, (n_seen, 1))
                part1 = 1.0 / probs_1o ** 2 - 1.0 / probs_1o
                part2 = 1.0 / (temp * temp.T) - 1.0 / probs_2o
                np.fill_diagonal(part2, 0.0)

                temp = np.tile(labels, (n_seen, 1))
                yi_yj = temp * temp.T

                var_1 = np.sum(part1 * labels) + np.sum(part2 * yi_yj)

            if stopping_condition is StoppingCondition.strict_v2:
                var_2 = (n_seen * labels / probs_1o - total_1o) ** 2
                var_2 = (n_total - n_seen) / n_total / n_seen / (n_seen - 1) * np.sum(var_2)

        n_seen_incl = labels.sum()

        if stopping_condition == StoppingCondition.loose:
            safe_to_stop = n_seen_incl >= recall_target * total_1o
        elif stopping_condition == StoppingCondition.strict_v1:
            safe_to_stop = n_seen_incl >= recall_target * (total_1o + np.sqrt(var_1))
        elif stopping_condition == StoppingCondition.strict_v2:
            safe_to_stop = n_seen_incl >= recall_target * (total_1o + np.sqrt(var_2))
        else:
            safe_to_stop = False

        return LogEntry(
            KEY=cls.KEY,
            safe_to_stop=safe_to_stop,
            score=n_seen_incl / total_1o,
            est_incl=total_1o,
            recall_target=recall_target,
            stopping_condition=stopping_condition,
            confidence_level=None,
        )


if __name__ == '__main__':
    from shared.test import test_method, plots

    params = MethodParams(
        stopping_condition=StoppingCondition.loose,
        recall_target=0.9,
    )
    dataset, results = test_method(AutoStop, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()

    params = MethodParams(
        stopping_condition=StoppingCondition.strict_v1,
        recall_target=0.9,
    )
    dataset, results = test_method(AutoStop, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()

    params = MethodParams(
        stopping_condition=StoppingCondition.strict_v2,
        recall_target=0.9,
    )
    dataset, results = test_method(AutoStop, params, 2)
    fig, ax = plots(dataset, results, params)
    fig.show()
