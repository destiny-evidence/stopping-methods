import logging
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from rankings import it_tuning_rankers as it_rankers
from shared.dataset import Dataset

logger = logging.getLogger('rank-simple')


def best_model_ranking(dataset: Dataset,
                       models: list[str],
                       repeat: int = 1,
                       train_proportion: float = 0.85,
                       random_state: int | None = None):
    # We'll store hyperparameters of the best model per batch here
    infos = []

    while dataset.has_unseen:
        logger.info(f'Running for batch {dataset.last_batch}...')
        dataset.prepare_next_batch()
        idxs = list(dataset.seen_data.index)
        train_idxs, test_idxs = train_test_split(
            idxs,
            train_size=train_proportion,
            stratify=dataset.seen_data['label'],
            random_state=random_state,
        )

        objectives = []
        scores = []
        params = []
        for ranker in it_rankers(models):
            ranker.attach_dataset(dataset)

            logger.info(f'Initializing and training ranker {ranker.name} '
                        f'for dataset {dataset.KEY} (run: {repeat}) '
                        f'on batch {dataset.last_batch} ({dataset.n_seen:,} / {dataset.n_total:,})...')
            ranker.init()
            ranker.train(idxs=train_idxs)

            logger.debug(f'Testing ranker {ranker.key} for dataset {dataset.KEY} '
                         f'on {1 - train_proportion:.0%} test set')
            predictions = ranker.predict(idxs=test_idxs)
            objective = recall_score(dataset.df.loc[test_idxs, 'label'] , predictions> 0.5)
            logger.debug(f'Recall is {objective:.2%} for ranker {ranker.key} on dataset {dataset.KEY} ')

            logger.debug('Predicting on all unseen...')
            predictions = ranker.predict(predict_on_all=False)

            objectives.append(objective)
            scores.append(predictions)
            params.append(ranker.get_params(preview=False))

        best_model = np.argmax(objectives)
        infos.append({
            'params': params[best_model],
            'recall': objectives[best_model],
            'batch_size': dataset.get_next_batch_size(),
        })
        dataset.register_predictions(scores=scores[best_model],
                                     model=params[best_model]['key'])
    return infos
