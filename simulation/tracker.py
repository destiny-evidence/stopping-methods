from typing import Any

import pandas as pd

from shared.config import settings
from shared.dataset import Dataset
from shared.method import AbstractLogEntry


class Tracker:
    def __init__(self):
        self.batch_i: int = 0
        self.batch_idxs: list[int] = []
        self.ranker: str | None = None
        self.dataset: Dataset | None = None
        self.log_entries: list[AbstractLogEntry] = []
        self.rows: list[dict[str, Any]] = []

    def register_batch(self, ranker: str, dataset: Dataset, batch_i: int, batch_idxs: list[int]) -> None:
        self.dataset = dataset
        self.batch_i = batch_i
        self.batch_idxs = batch_idxs
        self.ranker = ranker

    def log_entry(self, entry: AbstractLogEntry) -> None:
        self.log_entries.append(entry)

    def commit_batch(self):
        batch = self.dataset.df.iloc[self.batch_idxs]

        base_entry = {
            'dataset': self.dataset.KEY,
            'ranker': self.ranker,
            'batch_i': self.batch_i,
            'batch_idxs': self.batch_idxs,
            'n_total': self.dataset.n_total,
            'n_seen': self.dataset.n_seen,
            'n_unseen': self.dataset.n_unseen,
            'n_incl': self.dataset.n_incl,
            'n_incl_seen': self.dataset.get_seen_data()['labels'].sum(),
            'n_incl_batch': batch['labels'].sum(),
            'n_records_batch': batch.shape[0],
        }
        for entry in self.log_entries:
            self.rows.append({
                **base_entry,
                **{
                    f'{entry.KEY}-{k}': v
                    for k, v in entry.model_dump().items()
                },
            })

        # Write entire log to disk
        pd.DataFrame(self.rows).to_csv(settings.result_data_path / 'results.csv', index=False)

        # Reset our batch trackers
        self.log_entries = []
        self.batch_idxs = []
        self.dataset = None
        self.ranker = None
