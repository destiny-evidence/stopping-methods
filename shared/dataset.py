import random
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, key: str, labels: list[int], texts: list[str]):
        self.KEY = key
        self.labels = labels
        self.texts = texts

        self.df = pd.DataFrame({'labels': self.labels, 'texts': self.texts, 'scores': None, 'is_prioritised': None})
        self.n_incl = self.df['labels'].sum()

        self.ordering = np.arange(len(self.texts))
        self.n_seen = 0

    @property
    def n_total(self) -> int:
        return self.df.shape[0]

    @property
    def n_unseen(self):
        return self.n_total - self.n_seen

    @property
    def has_unseen(self):
        return self.n_unseen > 0

    def __len__(self) -> int:
        return self.n_total

    def shuffle_unseen(self):
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
        return idxs, list(batch['labels']), list(batch['texts'])

    def get_seen_data(self):
        return self.df[self.ordering[:self.n_seen]]

    def register_predictions(self, scores: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        if len(scores) != self.n_unseen:
            raise AttributeError('Prediction scores to not match number of remaining unseen documents')
        self.df.iloc[self.ordering[self.n_seen:], 'scores'] = scores
        self.ordering[self.n_seen:] = self.ordering[scores.argsort() + self.n_seen]
