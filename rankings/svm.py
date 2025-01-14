import json
import logging
from hashlib import sha1
from typing import Any

import nltk
import numpy as np
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from shared.dataset import Dataset
from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('rank-svm')


# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class SVMRanking(AbstractRanker):
    def __init__(self,
                 dataset: Dataset,
                 train_mode: TrainMode = TrainMode.RESET,
                 tuning: bool = False,
                 model_params: dict[str, Any] | None = None,
                 max_features: int = 75000,
                 ngram_range: tuple[int, int] = (1, 3),
                 min_df: int | float = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(dataset=dataset, train_mode=train_mode, tuning=tuning, **kwargs)
        self.model_params = {
            'degree': 3,
            'gamma': 'auto',
            'class_weight': 'balanced',
            'probability': True,
            **(model_params or {'C': 1.0, 'kernel': 'linear'})}

        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df,
                                          strip_accents='unicode')
        self.texts = None
        self.vectors = None
        self.model = None

    @property
    def key(self):
        key = (f'svm'
               f'-{self.train_mode}'
               f'-{self.dataset.batch_strategy}'
               f'-{self.vectorizer.ngram_range[0]}_{self.vectorizer.ngram_range[1]}'
               f'-{self.vectorizer.max_features}')
        if self.tuning:
            key = f'{key}-tuned'
        return f'{key}-{self.get_hash()}'

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
            parameters = {'C': [1, 10],
                          'gamma': [0.001, 0.01, 1],  # , 'auto', 'scale'
                          'kernel': ['linear', 'rbf']}  # , 'sigmoid'
            clf = SVC(**self.model_params)
            self.model = GridSearchCV(estimator=clf, param_grid=parameters, cv=StratifiedKFold(n_splits=2), refit=True)
        else:
            self.model = SVC(**self.model_params)

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
        if hasattr(self.model, 'C'):
            return {
                **base,
                'C': self.model.C,
                'gamma': self.model.gamma,
                'kernel': self.model.kernel,
            }
        return {
            **base,
            'C': self.model.best_estimator_.C,
            'gamma': self.model.best_estimator_.gamma,
            'kernel': self.model.best_estimator_.kernel,
        }
