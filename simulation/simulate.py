import logging
from enum import Enum

import typer

from rankings.svm import SVMRanking

from shared.config import settings
from shared.dataset import BatchStrategy
from shared.ranking import TrainMode
from simulation.tracker import Tracker
from simulation.iterators import it_methods, it_rankers, it_datasets

logger = logging.getLogger('simulation')

app = typer.Typer()


@app.command()
def produce_rankings(
        use_svm: bool = True,
        use_sdg: bool = True,
        num_repeats: int = 3,
        batch_strategy: BatchStrategy = BatchStrategy.DYNAMIC,
        stat_batch_size: int = 100,
        dyn_min_batch_incl: int = 5,
        dyn_min_batch_size: int = 100,
        dyn_growth_rate: float = 0.5,
        dyn_max_batch_size: int = 600,
        inject_random_batch_every: int = 0,
        predict_on_all: bool = True,  # if false, will only predict on unseen documents
):
    import nltk
    logger.debug('Loading NLTK data...')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    logger.info(f'Data path: {settings.DATA_PATH}')
    for repeat in range(1, num_repeats + 1):
        logger.info(f'Running for repeat cycle {repeat}...')
        for dataset in it_datasets():
            logger.info(f'Running simulation on dataset: {dataset.KEY}')
            logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total} '
                        f'=> {dataset.n_incl / dataset.n_total:.2%}')

            # override batch setup
            dataset.batch_strategy = batch_strategy
            dataset.batch_size = stat_batch_size
            dataset.min_batch_incl = dyn_min_batch_incl
            dataset.min_batch_size = dyn_min_batch_size
            dataset.growth_rate = dyn_growth_rate
            dataset.max_batch_size = dyn_max_batch_size
            dataset.inject_random_batch_every = inject_random_batch_every

            if use_svm:
                logger.info(f'Running SVM setups...')
                for ranker in [
                    SVMRanking(dataset=dataset, model_params={'C': 1.0, 'kernel': 'linear'}),
                    SVMRanking(dataset=dataset, model_params={'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}),
                    SVMRanking(dataset=dataset, tuning=True, train_mode=TrainMode.RESET),
                    SVMRanking(dataset=dataset, ngram_range=(1, 1), max_features=5000,
                               model_params={'C': 1.0, 'kernel': 'linear'}),
                ]:
                    target_key = f'{dataset.KEY}-{repeat}-{ranker.key}'
                    logger.info(f'Running ranker {target_key}...')
                    logger.debug(f'Checking for {settings.ranking_data_path / f'{target_key}.feather'}')
                    if ((settings.ranking_data_path / f'{target_key}.feather').exists()
                            or (settings.ranking_data_path / f'{target_key}.csv').exists()):
                        logger.info(f' > Skipping {target_key}; simulation already exists')
                        continue

                    ranker.init()

                    while dataset.has_unseen:
                        logger.info(f'Running for batch {dataset.last_batch}...')
                        dataset.prepare_next_batch()
                        ranker.train()
                        predictions = ranker.predict(predict_on_all=predict_on_all)
                        dataset.register_predictions(scores=predictions)

                    # persist to disk and reset
                    logger.info(f'Persisting to disk for {target_key}...')
                    ranker.store_info(settings.ranking_data_path,
                                      extra={
                                          'repeat': repeat,
                                      })
                    dataset.store(settings.ranking_data_path / f'{target_key}.feather')
                    dataset.store(settings.ranking_data_path / f'{target_key}.csv')
                    dataset.reset()


@app.command()
def evaluate_stopping(bla: str):
    print(bla)


# def full_run(strategy: BatchSizeStrategy,
#              min_batch_size: int = 100,
#              max_batch_size: int = 100):
#     tracker = Tracker()
#     logger.info(f'Data path: {settings.DATA_PATH}')
#     for dataset in it_datasets():
#         logger.info(f'Running simulation on dataset: {dataset.KEY}')
#         logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total}')
#         dataset.shuffle_unseen()
#         for ranker in it_rankers(dataset=dataset):
#             batch_i = -1
#             while dataset.has_unseen:
#                 batch_i += 1
#                 logger.info(f'Executing for batch {batch_i}')
#                 batch_size = get_batch_size(strategy=strategy,
#                                             min_batch_size=min_batch_size,
#                                             max_batch_size=max_batch_size,
#                                             batch_i=batch_i,
#                                             n_total=dataset.n_total)
#
#                 batch_idxs, batch_labels, _ = dataset.get_next_batch(batch_size=batch_size)
#                 logger.info(f'Batch idxs ({batch_size}): {batch_idxs}')
#                 logger.info(f'Batch labels {sum(batch_labels)} / {len(batch_labels)}')
#
#                 ranker.train()
#                 predictions = ranker.predict()
#                 dataset.register_predictions(scores=predictions)
#
#                 seen_df = dataset.get_seen_data()
#
#                 for method in it_methods(dataset=dataset):
#                     logger.info(f'Running method {method.KEY}')
#                     for paramset in method.parameter_options():
#                         tracker.log_entry(method.compute(
#                             list_of_labels=seen_df['labels'],
#                             list_of_model_scores=seen_df['scores'],
#                             is_prioritised=seen_df['is_prioritised'],
#                             **paramset)
#                         )
#
#                 tracker.commit_batch(model=ranker, dataset=dataset, batch_i=batch_i, batch_idxs=batch_idxs)


if __name__ == '__main__':
    app()
