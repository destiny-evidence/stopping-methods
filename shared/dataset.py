import random
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, labels: list[int], texts: list[str]):
        self.labels = labels
        self.texts = texts

        self.df = pd.DataFrame({'labels': self.labels, 'texts': self.texts})

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

    def shuffle_unseen(self):
        random.shuffle(self.ordering[self.n_seen:])

    def get_next_batch(self, batch_size: int) -> tuple[list[int], list[int], list[str]]:
        """
        :param batch_size:
        :return:
           list of ids in next batch,
           list of labels in next batch,
           list of texts in next batch
        """
        if self.n_seen >= self.n_total:
            raise StopIteration

        idxs = list(self.ordering[self.n_seen:self.n_seen + batch_size])
        batch = self.df.iloc[idxs]
        self.n_seen += batch_size
        return idxs, list(batch['labels']), list(batch['texts'])

    def get_training(self):
        pass

    def update_order(self, scores: np.ndarray[tuple[int], np.dtype[np.int_]]):
        if len(scores) != self.n_unseen:
            raise AttributeError('Prediction scores to not match number of remaining unseen documents')
        self.ordering[self.n_seen:] = self.ordering[scores.argsort() + self.n_seen]
