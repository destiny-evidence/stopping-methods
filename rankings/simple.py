import logging
from abc import abstractmethod
from typing import Any, Type

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('rank-simple')


class _SimpleRanking(AbstractRanker):
    def __init__(self,
                 BaseModel: Type[SGDClassifier | SVC | LogisticRegression],
                 model_params: dict[str, Any],
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 scoring: str | None = None,
                 tuning_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(train_mode=train_mode, tuning=tuning, **kwargs)
        self.model_params = model_params
        self.BaseModel = BaseModel
        self.scoring = scoring
        self.tuning_params = tuning_params
        self.random_seed = random_seed
        self.model = None
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

    def _init_model(self):
        if self.tuning:
            clf = self.BaseModel(**self.model_params)
            self.model = GridSearchCV(estimator=clf, param_grid=self.tuning_params, scoring=self.scoring,
                                      cv=StratifiedKFold(n_splits=2), refit=True)
        else:
            self.model = self.BaseModel(**self.model_params)

    def train(self, idxs: list[int] | None = None):
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
            self._init_model()
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
            'ngram_range': str(self.dataset.vectorizer.ngram_range),
            'max_features': self.dataset.vectorizer.max_features,
            'min_df': self.dataset.vectorizer.min_df,
            **self.model_params
        }
        if preview:
            return base
        if hasattr(self.model, 'best_estimator_'):
            return {
                **base,
                **{f'{self.name}-{k}': getattr(self.model.best_estimator_, k) for k in self.model_params.keys()}
            }
        return {
            **base,
            **{f'{self.name}-{k}': getattr(self.model, k) for k in self.model_params.keys()}
        }


class SVMRanker(_SimpleRanking):
    name = 'svm'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
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
                'max_iter': 1000,
                **(model_params or {})
            },
            train_mode=train_mode,
            tuning=tuning,
            tuning_params={
                'C': [1, 10],
                'gamma': [0.001, 0.01, 1],  # , 'auto', 'scale'
                'kernel': ['linear', 'rbf']  # , 'sigmoid'
            },
            scoring='recall',
            random_seed=random_seed,
            **kwargs)


class SGDRanker(_SimpleRanking):
    name = 'sgd'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 random_seed: int | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(
            BaseModel=SGDClassifier,
            model_params={
                'class_weight': 'balanced',
                'loss': 'log_loss',
                'max_iter': 1000,
                **(model_params or {})
            },
            train_mode=train_mode,
            tuning=tuning,
            tuning_params={
                'alpha': 10.0 ** -np.arange(1, 7)
            },
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )


class RegressionRanker(_SimpleRanking):
    name = 'logreg'

    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
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
                **(model_params or {})
            },
            train_mode=train_mode,
            tuning=tuning,
            tuning_params={
                'solver': ['saga', 'liblinear', 'lbfgs']  # 'newton-cholesky',
            },
            scoring='recall',
            random_seed=random_seed,
            **kwargs
        )
