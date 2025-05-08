import logging
from typing import Generator

import typer

from rankings.simple import SVMRanker, SDGRanker, RegressionRanker, LightGBMRanker
from rankings.simple import TrainMode
from shared.config import settings
from shared.dataset import BatchStrategy
from shared.ranking import AbstractRanker
from datasets import it_datasets

logger = logging.getLogger('precompute ranks')

app = typer.Typer()


def it_rankers(use_svm: bool = False,
               use_sdg: bool = False,
               use_reg: bool = False,
               use_lgbm: bool = False,
               use_fine_tuning: bool = False) -> Generator[AbstractRanker, None, None]:
    if use_reg or use_sdg or use_svm:
        import nltk
        logger.debug('Loading NLTK data...')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')

    if use_reg:
        logger.info('Using regression model...')
        yield RegressionRanker()
        # logger.info('Using regression with cholesky solver...')
        # yield RegressionRanker(model_params={'solver': 'newton-cholesky'})
        logger.info('Using regression with smaller input...')
        yield RegressionRanker(ngram_range=(1, 1), max_features=5000)

        if use_fine_tuning:
            logger.info('Using regression with tuning...')
            yield SDGRanker(tuning=True)

    if use_sdg:
        logger.info('Using SDG model...')
        yield SDGRanker()
        logger.info('Using SDG model with smaller input...')
        yield SDGRanker(ngram_range=(1, 1), max_features=5000)

        if use_fine_tuning:
            logger.info('Using SDG model with tuning...')
            yield SDGRanker(tuning=True)

    if use_svm:
        logger.info('Using SVM model...')
        yield SVMRanker(model_params={'C': 1.0, 'kernel': 'linear'})
        logger.info('Using SVM model with rbf...')
        yield SVMRanker(model_params={'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'})
        logger.info('Using SVM model with smaller input...')
        yield SVMRanker(model_params={'C': 1.0, 'kernel': 'linear'},
                        ngram_range=(1, 1), max_features=5000)
        
        if use_fine_tuning:
            logger.info('Using SVM model with tuning...')
            yield SVMRanker(tuning=True)
        
    if use_lgbm:
        logger.info('Using LightGBM model...')
        yield LightGBMRanker()

        if use_fine_tuning:
            logger.info('Using SVM model with tuning...')
            yield LightGBMRanker(tuning=True)


@app.command()
def produce_rankings(
        use_svm: bool = False,
        use_sdg: bool = False,
        use_reg: bool = False,
        use_lgbm: bool = False,
        use_fine_tuning: bool = False,
        num_repeats: int = 3,
        num_random_init: int = 100,
        batch_strategy: BatchStrategy = BatchStrategy.DYNAMIC,
        stat_batch_size: int = 100,
        dyn_min_batch_incl: int = 5,
        dyn_min_batch_size: int = 100,
        dyn_growth_rate: float = 0.1,
        dyn_max_batch_size: int = 600,
        min_incl_size: int = 10,
        min_dataset_size: int = 2000,
        inject_random_batch_every: int = 0,
        predict_on_all: bool = True,  # if false, will only predict on unseen documents
):
    logger.info(f'Data path: {settings.DATA_PATH}')
    settings.ranking_data_path.mkdir(parents=True, exist_ok=True)

    for dataset in it_datasets():
        logger.info(f'Running simulation on dataset: {dataset.KEY}')
        logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total} '
                    f'=> {dataset.n_incl / dataset.n_total:.2%}')
        
        if dataset.n_incl < min_incl_size:
            logger.warning(f' > Skipping {dataset.KEY}; too few included documents')
            continue
        if dataset.n_total < min_dataset_size:
            logger.warning(f' > Skipping {dataset.KEY}; too few documents')
            continue

        # override batch setup
        dataset.num_random_init = num_random_init
        dataset.batch_strategy = batch_strategy
        dataset.batch_size = stat_batch_size
        dataset.min_batch_incl = dyn_min_batch_incl
        dataset.min_batch_size = dyn_min_batch_size
        dataset.growth_rate = dyn_growth_rate
        dataset.max_batch_size = dyn_max_batch_size
        dataset.inject_random_batch_every = inject_random_batch_every

        logger.info(f'Running setups...')
        for ranker in it_rankers(use_svm=use_svm, use_reg=use_reg, use_sdg=use_sdg, use_lgbm=use_lgbm,
                                 use_fine_tuning=use_fine_tuning):
            for repeat in range(1, num_repeats + 1):
                logger.info(f'Running for repeat cycle {repeat}...')
                ranker.attach_dataset(dataset)
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
                ranker.store_info(settings.ranking_data_path / f'{target_key}.json',
                                  extra={
                                      'repeat': repeat,
                                  })
                dataset.store(settings.ranking_data_path / f'{target_key}.feather')
                # dataset.store(settings.ranking_data_path / f'{target_key}.csv')
                dataset.reset()
