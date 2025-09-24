import os
import logging
import warnings
from abc import abstractmethod
from typing import Any, Type

import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.base import clone as clone_model
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

from shared.config import settings
from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('rank-simple')
logging.getLogger('LightGBM').setLevel(logging.ERROR)

# Stop optuna from logging all trial results
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# Capturing some sklearn warnings to clear up logs
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore')
# Not everything is caught when parallelising, this helps...
os.environ['PYTHONWARNINGS'] = 'ignore'

type Classifier = SGDClassifier | SVC | LogisticRegression | LGBMClassifier | GaussianNB | IsolationForest


class _SimpleRanking(AbstractRanker):
    def __init__(self,
                 BaseModel: Type[Classifier],
                 model_params: dict[str, Any],
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 scoring: str | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(train_mode=train_mode, tuning=tuning, **kwargs)
        self.model_params = model_params
        self.BaseModel = BaseModel
        self.scoring = scoring
        self.tuning_trials = tuning_trials
        self.random_seed = random_seed
        self.model = None

    @abstractmethod
    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError()

    @property
    def key(self):
        key = (f'{self.name}'
               f'-{self.train_mode}'
               f'-{self.dataset.batch_strategy}'
               f'-{self.dataset.vectorizer.ngram_range[0]}_{self.dataset.vectorizer.ngram_range[1]}'
               f'-{self.dataset.vectorizer.max_features}')
        if self.tuning:
            key = f'{key}-tuned'
        return f'{key}-{self.get_hash()}'

    def init(self) -> None:
        pass

    def clear(self):
        self.model = None

    def _tune(self, trial: optuna.Trial, x: np.ndarray, y: np.ndarray) -> float:
        self.model_params = (self.model_params | self._hp_space(trial))
        model = self.BaseModel(**self.model_params)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_seed)
        score = cross_val_score(model, x, y, cv=cv, scoring=self.scoring)
        return score.mean()

    def train(self, idxs: list[int] | None = None, clone: bool = False) -> None:
        if not idxs:
            idxs = self.dataset.seen_data.index
        x = self.dataset.vectors[idxs]
        y = self.dataset.df.loc[idxs]['label']

        logger.debug(f'Fitting on {y.shape[0]:,} samples ({y.sum():,} of which included)')

        if self.train_mode == TrainMode.NEW:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.FULL:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.RESET:
            if self.tuning and self.model is not None and clone:
                self.model: Classifier = clone_model(self.model)
                self.model.fit(x, y)
            elif self.tuning:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self._tune(trial, x, y),
                               n_trials=self.tuning_trials,
                               n_jobs=settings.N_JOBS)
                logger.debug(f'Hyper-parameter-tuning for {self.name} done with best score {study.best_value}')
                self.model = self.BaseModel(**(self.model_params | study.best_params))
                self.model.fit(x, y)
            else:
                self.model = self.BaseModel(**self.model_params)
                self.model.fit(x, y)

    def predict(self, idxs: list[int] | None = None, predict_on_all: bool = True) -> np.ndarray:
        if not idxs:
            idxs = (self.dataset.unseen_data if not predict_on_all else self.dataset.df).index

        if len(idxs) == 0:
            return np.array([])

        x = self.dataset.vectors[idxs]
        y = self.dataset.df.loc[idxs]['label']

        logger.debug(f'Predicting on {y.shape[0]:,} samples ({y.sum():,} of which should be included)')
        y_preds = self.model.predict_proba(x)
        logger.debug(f'  > Predictions found {(y_preds > 0.5).sum():,} to be included')
        return y_preds[:, 1]

    def _get_params(self, preview: bool = True) -> dict[str, Any]:
        base = {
            'vectoriser': {
                'ngram_range': self.dataset.vectorizer.ngram_range,
                'max_features': self.dataset.vectorizer.max_features,
                'min_df': self.dataset.vectorizer.min_df,
            },
            'model': self.name,
        }
        if preview:
            return base | {
                'hyperparams': self.model_params
            }
        return base | {
            'hyperparams': {
                k: getattr(self.model, k)
                for k in self.model_params.keys()
            }
        }


class SVMRanker(_SimpleRanking):
    name = 'svm'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=SVC,
            model_params={
                             'kernel': 'linear',
                             'class_weight': 'balanced',
                             'degree': 3,
                             'gamma': 'auto',
                             'probability': True,
                             'C': 1.0,
                             'max_iter': 1000
                         } | (model_params or {}),
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            random_seed=random_seed,
            **kwargs)

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'C': trial.suggest_float('C', low=0.001, high=100, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf'])  # , 'poly', 'sigmoid'
        }


class SGDRanker(_SimpleRanking):
    name = 'sgd'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=SGDClassifier,
            model_params={
                             'class_weight': 'balanced',
                             'loss': 'log_loss',
                             'max_iter': 1000
                         } | (model_params or {}),
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'alpha': trial.suggest_float('alpha', 0.0001, 100000, log=True),
        }


class RegressionRanker(_SimpleRanking):
    name = 'logreg'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=LogisticRegression,
            model_params={
                             'class_weight': 'balanced',
                             'tol': 0.0001,
                             'C': 1.0,
                             'solver': 'lbfgs',
                             'max_iter': 100,
                         } | (model_params or {}),
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'C': trial.suggest_float('C', low=0.01, high=10, log=True),
            'solver': trial.suggest_categorical('solver', ['saga', 'liblinear', 'lbfgs']),
        }


class IsolationForestRanker(_SimpleRanking):
    name = 'isoforest'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=IsolationForest,
            model_params={
                             'n_estimators': 100,
                             'max_samples': "auto",
                             'contamination': "auto",
                             'max_features': 1.0,
                             'bootstrap': False,
                             'n_jobs': None,
                             'random_state': None,
                             'verbose': 0,
                             'warm_start': False,
                         } | (model_params or {}),
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'n_estimators': trial.suggest_float('n_estimators', low=20, high=250, log=True),
            'max_features': trial.suggest_float('max_features', low=0.2, high=1.0),
        }


class NaiveBayesRanker(_SimpleRanking):
    name = 'naivebayes'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=GaussianNB,
            model_params=(model_params or {}),
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {}


class LightGBMRanker(_SimpleRanking):
    name = 'lightgbm'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 tuning_trials: int = 35,
                 model_params: dict[str, Any] | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=LGBMClassifier,
            model_params={
                'objective': 'binary',  # (not multiclass))
                'learning_rate': 0.1,
                'n_estimators': 100,  # Number of boosting rounds
                'num_leaves': 31,  # Number of leaves in each tree
                'random_state': 42,  # For reproducibility
                'verbose': -1,
                **(model_params or {})
            },
            train_mode=train_mode,
            tuning=tuning,
            tuning_trials=tuning_trials,
            scoring='recall',
            **kwargs
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            # Controls step size in boosting
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            # Number of boosting rounds (discrete float, behaves like int)
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
            # Number of leaves in each tree (higher = more complex)
            'num_leaves': trial.suggest_int('num_leaves', 10, 50, log=True),
            # Depth of trees (-1 means no limit)
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            # Minimum data points in a leaf
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 20, log=True),
            # Fraction of samples used in each boosting iteration
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            # Fraction of features used per tree
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            # L1 regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            # L2 regularization
            'reg_lambda': trial.suggest_float('reg_alpha', 0.0, 1.0),
        }
