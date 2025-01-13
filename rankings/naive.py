import logging
from typing import Any, Literal

import nltk
import numpy as np
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight

from shared.dataset import Dataset
from shared.ranking import AbstractRanker

logger = logging.getLogger('naive rank')

type Variant = Literal['SVM', 'SGD']


class NaiveRankings(AbstractRanker):
    KEY: str = 'NAIVE'

    def __init__(self,
                 dataset: Dataset,
                 train_on_new_only: bool = False,
                 train_from_scratch: bool = True,
                 variant: Variant = 'SGD',
                 **kwargs: dict[str, Any]):
        super().__init__(dataset=dataset,
                         train_on_new_only=train_on_new_only,
                         train_from_scratch=train_from_scratch,
                         **kwargs)

        self.variant = variant

        stopwords = sw.words('english')

        logger.info('Preprocess texts')
        texts = dataset.texts
        self.texts = [
            ' '.join([tok
                      for tok in nltk.word_tokenize(text)
                      if tok not in stopwords])
            for text in texts
        ]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=75000, min_df=3, strip_accents='unicode')
        self.vectors = self.vectorizer.fit_transform(self.texts)

        self.model = None

    def train(self):
        seen = self.dataset.get_seen_data()
        x = self.vectors[seen.index]
        y = seen['label']

        if self.variant == 'SVM':
            # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight='balanced')

        elif self.variant == 'SGD':
            model = SGDClassifier(class_weight='balanced', loss='log_loss')
            parameters = {'alpha': 10.0 ** -np.arange(1, 7)}
            self.model = GridSearchCV(model, parameters, scoring='roc_auc', cv=StratifiedKFold(n_splits=2))

        else:
            raise NotImplementedError(f'Unknown variant {self.variant}')

        self.model.fit(x, y)

    def predict(self) -> np.ndarray:
        unseen = self.dataset.df.iloc[self.dataset.ordering[self.dataset.n_seen:]]
        if len(unseen) == 0:
            return np.array([])

        y_preds = self.model.predict_proba(self.vectors[unseen.index])
        return y_preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {'variant': self.variant, **self.model.get_params()}
