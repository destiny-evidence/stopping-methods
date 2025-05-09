import re
import logging
from typing import Any, Type

import numpy as np
from nltk import WordNetLemmatizer, pos_tag, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('rank-simple')

lemmatizer = WordNetLemmatizer()
NOALPH = re.compile(r'[^A-Za-z]+')

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    return lemmatizer.lemmatize(token, tag)


# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class _SimpleRanking(AbstractRanker):
    def __init__(self,
                 name: str,
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
            ' '.join([
                lemmatize(tok, tag)
                for text in texts
                for sentence in sent_tokenize(text)
                for tok, tag in pos_tag(wordpunct_tokenize(sentence))
                if tok not in stopwords and len(NOALPH.sub('', tok)) >= 3
            ])
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
