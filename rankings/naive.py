import logging
from typing import Any

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from shared.dataset import Dataset
from shared.ranking import AbstractRanker

logger = logging.getLogger('naive rank')

logger.debug('Loading NLTK data...')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def word_type(token: str):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


class NaiveRankings(AbstractRanker):
    def __init__(self,
                 dataset: Dataset,
                 train_on_new_only: bool = False,
                 train_from_scratch: bool = True,
                 **kwargs: dict[str, Any]):
        super().__init__(dataset, train_on_new_only, train_from_scratch, **kwargs)

        self.lemma = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')

        logger.info('Preprocess texts')
        texts = dataset['text']
        texts = [self.lemma.lemmatize(w, word_type(w)).lower()
                 for text in texts
                 for w in nltk.word_tokenize(text)]
        self.texts = [
            ' '.join([tok for tok in text if tok not in self.stopwords])
            for text in texts
        ]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.vectors = self.vectorizer.fit_transform(self.texts)

        self.model = None


def train(self):
    self.model()
    self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    seen = self.dataset.get_seen_data()
    self.model.fit(self.vectors[seen.index], seen[seen.index]['labels'])
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)


def predict(self) -> np.ndarray:
    pass
