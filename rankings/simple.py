import logging
from typing import Any, Type

import nltk
import numpy as np
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
#import lightgbm as lgb

from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('rank-simple')


# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class _SimpleRanking(AbstractRanker):
    def __init__(self,
                 name: str,
                 BaseModel: Type[SGDClassifier | SVC | LogisticRegression | LGBMClassifier],
                 model_params: dict[str, Any],
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 scoring: str | None = None,
                 tuning_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(train_mode=train_mode, tuning=tuning, **kwargs)
        self.model_params = model_params
        self.name = name
        self.BaseModel = BaseModel
        self.scoring = scoring
        self.tuning_params = tuning_params

        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df,
                                          strip_accents='unicode')
        self.texts = None
        self.vectors = None
        self.model = None

    @property
    def key(self):
        key = (f'{self.name}'
               f'-{self.train_mode}'
               f'-{self.dataset.batch_strategy}'
               f'-{self.vectorizer.ngram_range[0]}_{self.vectorizer.ngram_range[1]}'
               f'-{self.vectorizer.max_features}')
        if self.tuning:
            key = f'{key}-tuned'
        return f'{key}-{self.get_hash()}'

    def clear(self):
        self.texts = None
        self.vectors = None
        self.model = None
        self.vectorizer = TfidfVectorizer(ngram_range=self.vectorizer.ngram_range,
                                          max_features=self.vectorizer.max_features,
                                          min_df=self.vectorizer.min_df,
                                          strip_accents='unicode')

    def init(self) -> None:
        logger.info('Preprocess texts')
        stopwords = sw.words('english')
        texts = self.dataset.texts
        self.texts = [
            ' '.join([tok
                      for tok in nltk.word_tokenize(text)
                      if tok not in stopwords])
            for text in texts
        ]
        self.vectors = self.vectorizer.fit_transform(self.texts)

    def _init_model(self):
        if self.tuning:
            clf = self.BaseModel(**self.model_params)
            self.model = GridSearchCV(estimator=clf, param_grid=self.tuning_params, scoring=self.scoring,
                                      cv=StratifiedKFold(n_splits=2), refit=True)
        else:
            self.model = self.BaseModel(**self.model_params)

    def train(self):
        seen = self.dataset.seen_data
        x = self.vectors[seen.index]
        y = seen['label']

        logger.debug(f'Fitting on {y.shape[0]:,} samples ({y.sum():,} of which included)')

        if self.train_mode == TrainMode.NEW:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.FULL:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.RESET:
            self._init_model()
            self.model.fit(x, y)

    def predict(self, predict_on_all: bool = True) -> np.ndarray:
        idxs = (self.dataset.unseen_data if not predict_on_all else self.dataset.df).index

        if len(idxs) == 0:
            return np.array([])

        y_preds = self.model.predict_proba(self.vectors[idxs])
        return y_preds[:, 1]

    def _get_params(self, preview: bool = True) -> dict[str, Any]:
        base = {
            'ngram_range': str(self.vectorizer.ngram_range),
            'max_features': self.vectorizer.max_features,
            'min_df': self.vectorizer.min_df,
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
    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(
            name='svm',
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
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            **kwargs)


class SDGRanker(_SimpleRanking):
    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(
            name='sdg',
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
            scoring='roc_auc',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            **kwargs
        )


class RegressionRanker(_SimpleRanking):
    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(
            name='logreg',
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
            scoring='roc_auc',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            **kwargs
        )

class LightGBMRanker(_SimpleRanking):
    def __init__(self,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(
            name='lightgbm',
            BaseModel=LGBMClassifier,
            model_params={
                'objective': 'binary',  # (not multiclass))
                'learning_rate':0.1,
                'n_estimators':100,    # Number of boosting rounds
                'num_leaves':31,       # Number of leaves in each tree
                'random_state':42, # For reproducibility
                **(model_params or {})
            },
            train_mode=train_mode,
            tuning=tuning,
            tuning_params={
                "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Controls step size in boosting
                "n_estimators": [50, 100, 200, 500],  # Number of boosting rounds
                "num_leaves": [10, 20, 31, 50],  # Number of leaves in each tree (higher = more complex)
                "max_depth": [-1, 5, 10, 20],  # Depth of trees (-1 means no limit)
                "min_child_samples": [5, 10, 20],  # Minimum data points required in a leaf
                "subsample": [0.8, 0.9, 1.0],  # Fraction of samples used in each boosting iteration
                "colsample_bytree": [0.8, 0.9, 1.0],  # Fraction of features used per tree
                "reg_alpha": [0, 0.1, 0.5, 1],  # L1 regularization
                "reg_lambda": [0, 0.1, 0.5, 1]  # L2 regularization
            },
            scoring='roc_auc',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            **kwargs
        )
