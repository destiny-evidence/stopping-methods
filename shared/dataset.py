import json
import logging
import random
from enum import Enum
from pathlib import Path
from typing import Generator
from itertools import chain, batched

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from shared.config import settings

logger = logging.getLogger(__name__)


class Record(BaseModel):
    model_config = ConfigDict(extra='allow')

    # A unique identifier for each reference within the dataset. Example: 1, 2, 3, 4, 5. Missing Values: Not allowed
    id: int
    # Title of the reference. Missing Values: Represented as None.
    title: str | None = None
    # Abstract of the reference. Missing Values: Represented as  None.
    abstract: str | None = None
    # PubMed ID of the reference (if available). Missing Values: Represented as None.
    pubmed_id: str | None = None
    # OpenAlex identifier of the reference (if available). Missing values: Represented as None.
    openalex_id: str | None = None
    # Digital Object Identifier for the reference (if available). Missing Values: Allowed.
    doi: str | None = None
    # Keywords for the reference, separated by semi-colons. Example: "Keyword1; Keyword2; Keyword3". Missing Values: Represented as None.
    keywords: str | None = None
    # Publication year
    year: int | None = None

    # Label indicating inclusion/exclusion of the reference at title-abstract level screening. Values: {0, 1}.
    # Missing Values: Not allowed (must always have a value).
    label_abs: int
    # Label indicating some classification (e.g., full-text relevance). Values: {0, 1}. Missing Values: None.
    label_ft: int | None = None


class BatchStrategy(str, Enum):
    STATIC = 'STATIC'
    DYNAMIC = 'DYNAMIC'


class Dataset:
    def __init__(self,
                 key: str,
                 labels: list[int], texts: list[str],
                 num_random_init: int = 100,
                 batch_strategy: BatchStrategy = BatchStrategy.STATIC,
                 stat_batch_size: int = 100,
                 dyn_min_batch_incl: int = 2,
                 dyn_min_batch_size: int = 100,
                 dyn_growth_rate: float = 0.5,
                 dyn_max_batch_size: int = 200,
                 inject_random_batch_every: int = 0):
        self.KEY = key
        self.labels = labels
        self.texts = texts

        self.df = pd.DataFrame()
        self.reset()

        self.num_random_init = num_random_init
        self.batch_strategy = batch_strategy
        self.batch_size = stat_batch_size
        self.min_batch_incl = dyn_min_batch_incl
        self.min_batch_size = dyn_min_batch_size
        self.growth_rate = dyn_growth_rate
        self.max_batch_size = dyn_max_batch_size
        self.inject_random_batch_every = inject_random_batch_every

    @property
    def n_total(self) -> int:
        return self.df.shape[0]

    @property
    def n_incl(self) -> int:
        return self.df['label'].sum()

    @property
    def n_seen(self) -> int:
        return self.df['batch'].notna().sum()

    @property
    def n_unseen(self):
        return self.df['batch'].isna().sum()

    @property
    def has_unseen(self):
        return self.n_unseen > 0

    @property
    def last_batch(self) -> int:
        last = self.df['batch'].max()
        return 0 if np.isnan(last) else last

    def __len__(self) -> int:
        return self.n_total

    @property
    def seen_data(self):
        return self.df[self.df['batch'].notna()].sort_values(by='order')

    @property
    def unseen_data(self):
        return self.df[self.df['batch'].isna()]

    def reset(self) -> None:
        self.df = pd.DataFrame({
            'id': np.arange(len(self.texts)),
            'batch': None,
            'order': None,
            'random': None,
            'label': self.labels,
            'text': self.texts,
        })

    def get_next_batch_size(self) -> int:
        logger.info(f'Batch size compute for {self.n_seen:,} seen, {self.n_unseen:,} unseen, {self.n_total:,} total '
                    f'({self.seen_data['label'].sum():,} includes seen |'
                    f' {self.unseen_data['label'].sum():,} includes left)')
        if self.n_seen >= self.n_total:
            logger.info('Computed next batch size: 0 (reached end of dataset)')
            return 0

        if self.batch_strategy == BatchStrategy.STATIC:
            batch_size = min(self.batch_size, self.n_total - self.n_seen)
            logger.info(f'Computed next batch size: {batch_size:,} (static {self.batch_size:,})')
            return batch_size

        if self.batch_strategy == BatchStrategy.DYNAMIC:
            if self.min_batch_incl > 0:
                remaining_includes = np.argwhere(self.unseen_data
                                                 .sort_values(by='order')['label']
                                                 .cumsum() > self.min_batch_incl)
                target = remaining_includes.min() if len(remaining_includes) > 0 else self.n_unseen
                logger.info(f'Computed target batch size: {target:,} (adaptive min. num. includes)')
            else:
                target = int(self.n_seen + (self.n_seen * self.growth_rate))
                logger.info(f'Computed target batch size: {target:,} (adaptive growth_rate @ {self.growth_rate})')

            batch_size = min(self.max_batch_size, max(self.min_batch_size, target))
            logger.info(f'Computed next batch size: {batch_size:,} '
                        f'(adaptive [{self.min_batch_size:,}, {self.max_batch_size:,}])')
            return batch_size

        raise AttributeError('Batch strategy not supported')

    def get_random_unseen_sample(self, sample_size: int | None = None) -> list[int]:
        logger.info('Preparing random sample')
        idxs = self.unseen_data.index.tolist()
        random.shuffle(idxs)
        batch_size = self.get_next_batch_size() if sample_size is None else sample_size
        if self.last_batch == 0:
            min_incl = self.min_batch_incl or 2
            num_incl = self.df.loc[idxs[:batch_size]]['label'].sum()

            if self.unseen_data['label'].sum() < min_incl:
                raise AssertionError('Not enough includes for initial random sample from unseen data!')

            if num_incl < min_incl:
                logger.warning('Initial sample did not have enough includes, going to inject some!')
                incl_idxs = self.unseen_data[self.unseen_data['label'] == 1].index.tolist()[:min_incl-num_incl]
                idxs = incl_idxs + idxs

        idxs = idxs[:batch_size]
        random.shuffle(idxs)
        return idxs

    def prepare_next_batch(self) -> None:
        if self.n_seen >= self.n_total:
            raise StopIteration

        # Handle random batch injection (or initial batch)
        if (self.last_batch == 0
                or (self.inject_random_batch_every > 0
                    and (self.last_batch % self.inject_random_batch_every) == 0)):
            idxs = self.get_random_unseen_sample()
            # add our batch data to the table
            batch_i = self.last_batch + 1
            n_seen = self.n_seen
            self.df.loc[self.df.iloc[idxs].index, 'order'] = np.arange(len(idxs)) + n_seen
            self.df.loc[self.df.iloc[idxs].index, 'batch'] = batch_i
            self.df.loc[self.df.iloc[idxs].index, 'random'] = True
            self.df[f'scores_batch_{batch_i}'] = -1

            # immediately continue with next batch
            self.prepare_next_batch()

    def register_predictions(self, scores: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        if len(scores) == 0:
            logger.warning('Tried to register predictions but scores are empty')
            return

        # fetch numbers here to avoid side effects later
        batch_i = self.last_batch + 1
        n_seen = self.n_seen
        logger.info(f'Registering predictions for {scores.shape} records as batch {batch_i}')

        batch_size = self.get_next_batch_size()

        if len(scores) == self.n_unseen:
            self.df.loc[self.unseen_data.index, f'scores_batch_{batch_i}'] = scores
        elif len(scores) == self.n_total:
            self.df[f'scores_batch_{batch_i}'] = scores
        else:
            raise AttributeError('Number of prediction scores do not match number of '
                                 'all or remaining unseen documents.')
        self.df.fillna({f'scores_batch_{batch_i}': -1}, inplace=True)
        ordering = (-self.unseen_data[f'scores_batch_{batch_i}']).argsort()
        self.df.loc[ordering.index, 'order'] = ordering + n_seen
        idxs = ordering.sort_values().index.to_list()[:batch_size]

        self.df.loc[idxs, 'batch'] = batch_i
        self.df.loc[idxs, 'random'] = False

    def store(self, target: Path) -> None:
        df = self.df.sort_values(by='order').drop('text', axis='columns')
        if target.suffix == '.csv':
            df.to_csv(target, index=False, float_format='%1.6f')
        elif target.suffix == 'arrow' or target.suffix == '.feather':
            df.to_feather(target)
        else:
            raise AttributeError(f'Unsupported file type {target.suffix}')


class RankedDataset:
    def __init__(self, ranking_info_fp: Path):
        self.ranking_fp = f'{ranking_info_fp.with_suffix('')}.feather'

        logger.info(f'Ranking from: {self.ranking_fp}')
        logger.debug(f'Info from {ranking_info_fp}')

        with open(ranking_info_fp, 'r') as f:
            self.info = json.load(f)
        self.ranking = pd.read_feather(self.ranking_fp)

    @property
    def n_total(self) -> int:
        return self.ranking.shape[0]

    @property
    def n_incl(self) -> int:
        return self.ranking['label'].sum()

    @property
    def n_batches(self) -> int:
        return len(self.ranking['batch'].unique())

    @property
    def batch_sizes(self) -> int:
        return self.ranking['batch'].value_counts()

    @property
    def inclusion_rate(self) -> float:
        return self.n_incl / self.n_total

    @property
    def dataset(self) -> str:
        return self.info['dataset']

    @property
    def repeat(self) -> str:
        return self.info['repeat']

    @property
    def ranker(self) -> str:
        return self.info['ranker']

    @property
    def __len__(self) -> int:
        return self.n_total

    @property
    def data(self) -> tuple[list[int], list[int], list[float], list[bool]]:
        batches = []
        scores = []
        labels = []
        is_prioritised = []
        for bi, b_labels, b_scores, b_prio in self.it_sim_batches():
            scores += b_scores
            labels += b_labels
            is_prioritised += b_prio
            batches += [bi] * len(b_scores)
        return batches, labels, scores, is_prioritised

    def it_sim_batches(self) -> Generator[tuple[int, list[int], list[float], list[bool]], None, None]:
        for bi, batch in self.ranking.groupby('batch'):
            batch = batch.sort_values('order')
            yield bi, batch['label'].tolist(), batch[f'scores_batch_{bi}'].tolist(), (~batch['random']).tolist()

    def it_cum_sim_batches(self) -> Generator[tuple[int, list[int], list[float], list[bool]], None, None]:
        scores = np.array([])
        labels = np.array([])
        is_prioritised = np.array([])
        for bi, b_scores, b_labels, b_prio in self.it_sim_batches():
            scores = np.concatenate(scores, b_scores)
            labels = np.concatenate(labels, b_labels)
            is_prioritised = np.concatenate(is_prioritised, b_prio)
            yield bi, labels, scores, is_prioritised

    def it_batches(self, batch_size: int) -> Generator[tuple[int, list[int], list[float], list[bool]], None, None]:
        for batches, labels, scores, is_prioritised in zip(*[batched(pt, batch_size) for pt in self.data]):
            yield batches, labels, scores, is_prioritised

    def it_cum_batches(self, batch_size: int) -> Generator[tuple[np.array, np.array, np.array, np.array], None, None]:
        batches = []
        scores = []
        labels = []
        is_prioritised = []
        for bis, b_labels, b_scores, b_prio in zip(*[batched(pt, batch_size) for pt in self.data]):
            batches += bis
            scores += b_scores
            labels += b_labels
            is_prioritised += b_prio
            yield np.array(batches), np.array(labels), np.array(scores), np.array(is_prioritised)
