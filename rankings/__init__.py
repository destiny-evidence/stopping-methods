import logging
from typing import Type, Generator

from shared.ranking import AbstractRanker, TrainMode
from .simple import SGDRanker, RegressionRanker, SVMRanker, LightGBMRanker
from .transformer import TransRanker

logger = logging.getLogger('ranker')

MODELS: dict[str, Type[AbstractRanker]] = {
    SGDRanker.name: SGDRanker,
    RegressionRanker.name: RegressionRanker,
    SVMRanker.name: SVMRanker,
    LightGBMRanker.name: LightGBMRanker,
    TransRanker.name: TransRanker,
}


def assert_models(models: list[str]) -> list[str]:
    if models is None:
        return list(MODELS.keys())
    else:
        for model in models:
            if model not in MODELS:
                raise ValueError(f'Model {model} is not supported')
        return models


def it_tuning_rankers(models: list[str] | None = None) -> Generator[AbstractRanker, None, None]:
    for model in (models or MODELS.keys()):
        yield MODELS[model](tuning=True, train_mode=TrainMode.RESET)


def it_rankers(models: list[str], use_fine_tuning: bool = False) -> Generator[AbstractRanker, None, None]:
    if RegressionRanker.name in models:
        logger.info('Using regression model...')
        yield RegressionRanker()
        logger.info('Using regression with cholesky solver...')
        yield RegressionRanker(model_params={'solver': 'newton-cholesky'})

        if use_fine_tuning:
            logger.info('Using regression with tuning...')
            yield SGDRanker(tuning=True)

    if SGDRanker.name in models:
        logger.info('Using SGD model...')
        yield SGDRanker()

        if use_fine_tuning:
            logger.info('Using SGD model with tuning...')
            yield SGDRanker(tuning=True)

    if SVMRanker.name in models:
        logger.info('Using SVM model...')
        yield SVMRanker(model_params={'C': 1.0, 'kernel': 'linear'})
        logger.info('Using SVM model with rbf...')
        yield SVMRanker(model_params={'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'})
        logger.info('Using SVM model with smaller input...')

        if use_fine_tuning:
            logger.info('Using SVM model with tuning...')
            yield SVMRanker(tuning=True)

    if LightGBMRanker.name in models:
        logger.info('Using LightGBM model...')
        yield LightGBMRanker()

        if use_fine_tuning:
            logger.info('Using SVM model with tuning...')
            yield LightGBMRanker(tuning=True)

# import rankings
# def it_rankers() -> Generator[AbstractRanker, None, None]:
#     for Ranker in map(rankings.__dict__.get, rankings.__all__):
#         yield Ranker()
