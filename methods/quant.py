from typing import Generator, TypedDict
import numpy as np
import pandas as pd
from shared.method import AbstractMethod, AbstractLogEntry
from shared.types import IntList, FloatList

# This is based on the implementation of the Quant Rule from the paper: Heuristic stopping rules for technology-assisted review (Yang 2021)
# Implementation taken from class QuantStoppingRule(StoppingRule) in Tarexo framework.


class QuantParamSet(TypedDict):
    target_recall: float
    nstd: float

class QuantLogEntry(AbstractLogEntry):
    KEY: str = 'QUANT'
    target_recall: float
    nstd: float
    est_recall: float
    est_var: float | None = None

class Quant(AbstractMethod):
    KEY: str = 'QUANT'
    
    def parameter_options(self) -> Generator[QuantParamSet, None, None]:
        for target_recall in [0.7, 0.8, 0.9, 0.95]:
            for nstd in [0, 1, 2]:
                yield QuantParamSet(target_recall=target_recall, nstd=nstd)
    
    def compute(self,
                list_of_labels: IntList,
                list_of_model_scores: FloatList,
                is_prioritised: list[int] | list[bool] | pd.Series | np.ndarray,
                target_recall: float = 0.9,
                nstd: float = 0) -> QuantLogEntry:
        
        if len(list_of_labels) < 2:
            return QuantLogEntry(
                safe_to_stop=False,
                target_recall=target_recall,
                nstd=nstd,
                est_recall=0.0
            )
        
        scores = np.array(list_of_model_scores)
        labels = np.array(list_of_labels)
        assert (scores <= 1).all() and (scores >= 0).all(), \
            "Scores have to be probabilities to use Quant Rule."
        
        annotated = labels == 1
        unknown_ps = scores[~annotated].sum()
        known_ps = scores[annotated].sum()
        est_recall = known_ps / (known_ps + unknown_ps)
        
        if nstd == 0:  
            return QuantLogEntry(
                safe_to_stop=est_recall >= target_recall,  
                target_recall=target_recall,  
                nstd=nstd,  
                est_recall=float(est_recall)
            )
        
        prod = scores * (1-scores)
        all_var = prod.sum()
        unknown_var = prod[~annotated].sum()
        est_var = (known_ps**2 / (known_ps + unknown_ps)**4 * all_var) + (1 / (known_ps + unknown_ps)**2 * (all_var-unknown_var))
        safe_to_stop = est_recall - nstd * np.sqrt(est_var) >= target_recall  
        return QuantLogEntry(
            safe_to_stop=safe_to_stop,
            target_recall=target_recall,  
            nstd=nstd,  
            est_recall=float(est_recall),
            est_var=float(est_var)
        )