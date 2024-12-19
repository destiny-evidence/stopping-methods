import logging
import random
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, key: str, labels: list[int], texts: list[str]):
        self.KEY = key
        self.labels = labels
        self.texts = texts

        self.df = pd.DataFrame({'labels': self.labels, 'texts': self.texts, 'scores': None, 'is_prioritised': None})

        self.ordering = np.arange(len(self.texts))
        self.n_seen = 0

    @property
    def n_total(self) -> int:
        return self.df.shape[0]

    @property
    def n_incl(self) -> int:
        return self.df['labels'].sum()

    @property
    def n_unseen(self):
        return self.n_total - self.n_seen

    @property
    def has_unseen(self):
        return self.n_unseen > 0

    def __len__(self) -> int:
        return self.n_total

    def shuffle_unseen(self):
        if self.n_seen == 0:
            logger.debug('Initial shuffle')
            random.shuffle(self.ordering)
            while self.ordering[:10].sum() == 0:
                logger.debug('Initial reshuffle to get some positives in the first batch')
                random.shuffle(self.ordering)
        else:
            random.shuffle(self.ordering[self.n_seen:])

    def get_next_batch(self, batch_size: int) -> tuple[list[int], list[int], list[str]]:
        """
        :param batch_size:
        :return:
           list of idxs in next batch,
           list of labels in next batch,
           list of texts in next batch
        """
        if self.n_seen >= self.n_total:
            raise StopIteration

        idxs = list(self.ordering[self.n_seen:self.n_seen + batch_size])
        batch = self.df.iloc[idxs]
        self.n_seen += batch_size
        return idxs, batch['labels'].to_list(), batch['texts'].to_list()

    def get_seen_data(self):
        return self.df.iloc[self.ordering[:self.n_seen]]

    def register_predictions(self, scores: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        logger.debug(f'Registering predictions for {scores.shape} scores in {self.ordering[self.n_seen:].shape}')
        if len(scores) != self.n_unseen:
            raise AttributeError('Prediction scores do not match number of remaining unseen documents.')
        if len(scores) == 0:
            return

        self.df.loc[self.df.iloc[self.ordering[self.n_seen:]].index, 'scores'] = scores
        self.ordering[self.n_seen:] = self.ordering[(-scores).argsort() + self.n_seen]
