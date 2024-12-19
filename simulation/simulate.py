import logging
from enum import Enum

import typer

from shared.config import settings
from simulation.tracker import Tracker
from simulation.iterators import it_methods, it_rankers, it_datasets


class BatchSizeStrategy(str, Enum):
    FIXED = 'FIXED'
    ADAPTIVE = 'ADAPTIVE'


logger = logging.getLogger('simulation')


def get_batch_size(strategy: BatchSizeStrategy,
                   batch_i: int,
                   n_total: int,
                   min_batch_size: int = 100,
                   max_batch_size: int = 100) -> int:
    if strategy == 'FIXED':
        return min_batch_size
    if strategy == 'ADAPTIVE':
        return min(min_batch_size * (batch_i + 1), max_batch_size)
    raise ValueError('Unknown strategy!')


def full_run(strategy: BatchSizeStrategy,
             min_batch_size: int = 100,
             max_batch_size: int = 100):
    tracker = Tracker()
    logger.info(f'Data path: {settings.DATA_PATH}')
    for dataset in it_datasets():
        logger.info(f'Running simulation on dataset: {dataset.KEY}')
        logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total}')
        dataset.shuffle_unseen()
        for ranker in it_rankers(dataset=dataset):
            batch_i = -1
            while dataset.has_unseen:
                batch_i += 1
                logger.info(f'Executing for batch {batch_i}')
                batch_size = get_batch_size(strategy=strategy,
                                            min_batch_size=min_batch_size,
                                            max_batch_size=max_batch_size,
                                            batch_i=batch_i,
                                            n_total=dataset.n_total)

                batch_idxs, batch_labels, _ = dataset.get_next_batch(batch_size=batch_size)
                logger.info(f'Batch idxs ({batch_size}): {batch_idxs}')
                logger.info(f'Batch labels {sum(batch_labels)} / {len(batch_labels)}')


                ranker.train()
                predictions = ranker.predict()
                dataset.register_predictions(scores=predictions)

                seen_df = dataset.get_seen_data()

                for method in it_methods(dataset=dataset):
                    for paramset in method.parameter_options():
                        tracker.log_entry(method.compute(
                            list_of_labels=seen_df['labels'],
                            list_of_model_scores=seen_df['scores'],
                            is_prioritised=seen_df['is_prioritised'],
                            **paramset)
                        )

                tracker.commit_batch(model=ranker, dataset=dataset, batch_i=batch_i, batch_idxs=batch_idxs)


if __name__ == '__main__':
    typer.run(full_run)
