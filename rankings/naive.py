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

logger.debug('Loading NLTK data...')
nltk.download('stopwords')
nltk.download('punkt')

type Variant = Literal['SVM', 'SGD']


class NaiveRankings(AbstractRanker):
    def __init__(self,
                 dataset: Dataset,
                 train_on_new_only: bool = False,
                 train_from_scratch: bool = True,
                 variant: Variant = 'SVM',
                 **kwargs: dict[str, Any]):
        super().__init__(dataset, train_on_new_only, train_from_scratch, **kwargs)
        self.variant = variant

        stopwords = sw.words('english')

        logger.info('Preprocess texts')
        texts = dataset.df['text']
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
        y = seen[seen.index]['labels']

        if self.variant == 'SVM':
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
            self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight=class_weights)

        elif self.variant == 'SGD':
            model = SGDClassifier(class_weight="balanced", loss="log")
            parameters = {'alpha': 10.0 ** -np.arange(1, 7)}
            self.model = GridSearchCV(model, parameters, scoring="roc_auc", cv=StratifiedKFold(n_splits=2))

        else:
            raise NotImplementedError(f'Unknown variant {self.variant}')

        self.model.fit(x, y)

    def predict(self) -> np.ndarray:
        unseen = self.dataset.df[self.dataset.ordering[:self.dataset.n_seen]]
        y_preds = self.model.predict_proba(self.vectors[unseen.index])
        return y_preds[:, 1]
