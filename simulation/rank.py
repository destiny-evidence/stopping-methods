import logging
from typing import Annotated

import typer
from rankings import assert_models, it_rankers
from rankings.use_best import best_model_ranking as bm_ranking
from shared.config import settings
from shared.dataset import BatchStrategy
from datasets import it_datasets
from shared.disk import json_dumps

logger = logging.getLogger('precompute ranks')

app = typer.Typer()


def prepare_nltk():
    logger.debug('Loading NLTK data...')
    from nltk import download

    download('stopwords')
    download('punkt')
    download('punkt_tab')
    download('wordnet')
    download('averaged_perceptron_tagger_eng')


@app.command()
def produce_rankings(
        models: list[str] | None = None,
        use_fine_tuning: bool = False,
        num_repeats: int = 3,
        num_random_init: int = 100,
        batch_strategy: BatchStrategy = BatchStrategy.DYNAMIC,
        stat_batch_size: int = 100,
        dyn_min_batch_incl: int = 5,
        dyn_min_batch_size: int = 100,
        dyn_growth_rate: float = 0.1,
        dyn_max_batch_size: int = 600,
        min_dataset_size: int = 500,
        min_inclusion_rate: float = 0.01,
        inject_random_batch_every: int = 0,
        max_vocab: int = 7000,
        max_ngram: int = 1,
        min_df: int = 3,
        store_feather: bool = True,
        store_csv: bool = False,
        predict_on_all: bool = True,  # if false, will only predict on unseen documents
):
    if min_dataset_size <= num_random_init:
        raise ValueError('min_dataset_size must be higher than num_random_init')

    logger.info(f'Data path: {settings.DATA_PATH}')
    settings.ranking_data_path.mkdir(parents=True, exist_ok=True)

    models = assert_models(models)
    prepare_nltk()

    for dataset in it_datasets():
        logger.info(f'Running simulation on dataset: {dataset.KEY}')
        logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total} '
                    f'=> {dataset.n_incl / dataset.n_total:.2%}')

        if dataset.n_total < min_dataset_size:
            logger.warning(f'Dataset {dataset.KEY} is too small {dataset.n_total} < {min_dataset_size}')
            continue
        if (dataset.n_incl / dataset.n_total) < min_inclusion_rate:
            logger.warning(f'Dataset {dataset.KEY} inclusion rate too small!')
            continue

        dataset.init(num_random_init=num_random_init, batch_strategy=batch_strategy,
                     stat_batch_size=stat_batch_size, dyn_min_batch_size=dyn_min_batch_size,
                     dyn_max_batch_size=dyn_max_batch_size, inject_random_batch_every=inject_random_batch_every,
                     dyn_min_batch_incl=dyn_min_batch_incl, dyn_growth_rate=dyn_growth_rate,
                     ngram_range=(1, max_ngram), max_features=max_vocab, min_df=min_df)

        for ranker in it_rankers(models=models, use_fine_tuning=use_fine_tuning):
            for repeat in range(1, num_repeats + 1):
                logger.info(f'Running for repeat cycle {repeat}...')
                ranker.attach_dataset(dataset)
                target_key = f'{dataset.KEY}-{repeat}-{ranker.key}'
                logger.info(f'Running ranker {target_key}...')
                logger.debug(f'Checking for {settings.ranking_data_path / f'{target_key}.json'}')
                if (settings.ranking_data_path / f'{target_key}.json').exists():
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
                if store_feather:
                    dataset.store(settings.ranking_data_path / f'{target_key}.feather')
                if store_csv:
                    dataset.store(settings.ranking_data_path / f'{target_key}.csv')
                dataset.reset()


@app.command()
def best_model_ranking(
        models: Annotated[list[str] | None, typer.Option(help='Models to use')] = None,
        num_repeats: int = 3,
        min_dataset_size: int = 500,
        min_inclusion_rate: float = 0.01,
        num_random_init: int = 100,
        batch_strategy: BatchStrategy = BatchStrategy.DYNAMIC,
        stat_batch_size: int = 100,
        dyn_min_batch_incl: int = 5,
        dyn_min_batch_size: int = 100,
        dyn_growth_rate: float = 0.1,
        dyn_max_batch_size: int = 600,
        inject_random_batch_every: int = 0,
        train_proportion: float = 0.85,
        max_vocab: int = 7000,
        max_ngram: int = 1,
        min_df: int = 3,
        tuning_interval: int = 1,
        random_state: int | None = None,
        store_feather: bool = True,
        store_csv: bool = False,
):
    logger.info(f'Data path: {settings.DATA_PATH}')
    settings.ranking_data_path.mkdir(parents=True, exist_ok=True)

    if min_dataset_size <= num_random_init:
        raise ValueError('min_dataset_size must be higher than num_random_init')

    models = assert_models(models)
    prepare_nltk()

    for dataset in it_datasets():
        logger.info(f'Running simulation on dataset: {dataset.KEY}')
        logger.info(f'  n_incl={dataset.n_incl}, n_total={dataset.n_total} '
                    f'=> {dataset.n_incl / dataset.n_total:.2%}')

        if dataset.n_total < min_dataset_size:
            logger.warning(f'Dataset {dataset.KEY} is too small {dataset.n_total} < {min_dataset_size}')
            continue
        if (dataset.n_incl / dataset.n_total) < min_inclusion_rate:
            logger.warning(f'Dataset {dataset.KEY} inclusion rate too small!')
            continue

        for repeat in range(1, num_repeats + 1):
            logger.info(f'Running for repeat cycle {repeat}...')

            target_key = f'{dataset.KEY}-{num_random_init}-{repeat}-best'
            logger.info(f'Running ranker {target_key}...')
            logger.debug(f'Checking for {settings.ranking_data_path / f'{target_key}.json'}')
            if ((settings.ranking_data_path / f'{target_key}.json').exists()):
                logger.info(f' > Skipping {target_key}; simulation already exists')
                continue

            if repeat == 1:
                dataset.init(num_random_init=num_random_init, batch_strategy=batch_strategy,
                             stat_batch_size=stat_batch_size, dyn_min_batch_size=dyn_min_batch_size,
                             dyn_max_batch_size=dyn_max_batch_size, inject_random_batch_every=inject_random_batch_every,
                             dyn_min_batch_incl=dyn_min_batch_incl, dyn_growth_rate=dyn_growth_rate,
                             ngram_range=(1, max_ngram), max_features=max_vocab, min_df=min_df)

            infos = bm_ranking(dataset=dataset,
                               models=models,
                               repeat=repeat,
                               train_proportion=train_proportion,
                               tuning_interval=tuning_interval,
                               random_state=random_state)

            # persist to disk and reset
            logger.info(f'Persisting to disk for {target_key}...')
            json_dumps(settings.ranking_data_path / f'{target_key}.json', {
                'repeat': repeat,
                'batches': infos,
            }, indent=2)

            if store_feather:
                dataset.store(settings.ranking_data_path / f'{target_key}.feather')
            if store_csv:
                dataset.store(settings.ranking_data_path / f'{target_key}.csv')

            dataset.reset()
