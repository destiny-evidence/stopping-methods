from typing import Any

import pandas as pd

from shared.config import settings
from shared.dataset import Dataset
from shared.method import AbstractLogEntry
from shared.ranking import AbstractRanker


class Tracker:
    def __init__(self):
        self.log_entries: list[AbstractLogEntry] = []
        self.rows: list[dict[str, Any]] = []

    def log_entry(self, entry: AbstractLogEntry) -> None:
        self.log_entries.append(entry)

    def commit_batch(self, model: AbstractRanker, dataset: Dataset, batch_i: int, batch_idxs: list[int]):
        batch = dataset.df.iloc[batch_idxs]

        base_entry = {
            'dataset': dataset.KEY,
            'ranker': model.KEY,
            'batch_i': batch_i,
            'batch_idxs': batch_idxs,
            'n_total': dataset.n_total,
            'n_seen': dataset.n_seen,
            'n_unseen': dataset.n_unseen,
            'n_incl': dataset.n_incl,
            'n_incl_seen': dataset.get_seen_data()['labels'].sum(),
            'n_incl_batch': batch['labels'].sum(),
            'n_records_batch': batch.shape[0],
        }
        for entry in self.log_entries:
            self.rows.append({
                **base_entry,
                'method': entry.KEY,
                'safe_to_stop': entry.safe_to_stop,
                **{
                    f'method-{entry.KEY}-{k}': v
                    for k, v in entry.model_dump().items()
                },
                **{
                    f'ranker-{k}': v
                    for k, v in model.get_params().items()
                }
            })

        # Write entire log to disk
        pd.DataFrame(self.rows).to_csv(settings.result_data_path / 'results.csv', index=False)

        # Reset our batch trackers
        self.log_entries = []
