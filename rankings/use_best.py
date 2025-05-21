import logging
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

from rankings import it_tuning_rankers as it_rankers
from shared.dataset import Dataset

logger = logging.getLogger('rank-simple')


def best_model_ranking(dataset: Dataset,
                       models: list[str],
                       repeat: int = 1,
                       train_proportion: float = 0.85,
                       tuning_interval: int = 1,
                       random_state: int | None = None) -> list[dict]:
    # We'll store hyperparameters of the best model per batch here
    infos = []

    last_ranker = None
    while dataset.has_unseen:
        dataset.prepare_next_batch()

        if (dataset.last_batch % tuning_interval) > 0 and last_ranker is not None:
            logger.info(f'Training without hyperparameter tuning in batch {dataset.last_batch}')
            last_ranker.train(clone=True)
            y_pred = last_ranker.predict(predict_on_all=False)

            infos.append({
                'params': last_ranker.get_params(preview=False),
                'batch_size': dataset.get_next_batch_size(),
                'batch_i': dataset.last_batch,
            })
            dataset.register_predictions(scores=y_pred,
                                         model=last_ranker.name)
        else:
            logger.info(f'Using hyperparameter tuning in batch {dataset.last_batch}')
            idxs = list(dataset.seen_data.index)
            train_idxs, test_idxs = train_test_split(
                idxs,
                train_size=train_proportion,
                stratify=dataset.seen_data['label'],
                random_state=random_state,
            )

            best_prec = 0
            best_rec = 0
            best_f1 = 0
            y_pred = None
            best_ranker = None
            for ranker in it_rankers(models):
                ranker.attach_dataset(dataset)
                logger.info(f'Initializing and training ranker {ranker.name} '
                            f'for dataset {dataset.KEY} (run: {repeat}) '
                            f'on batch {dataset.last_batch} ({dataset.n_seen:,} / {dataset.n_total:,})...')
                ranker.init()
                ranker.train(idxs=train_idxs)

                logger.debug(f'Testing ranker {ranker.name} for dataset {dataset.KEY} '
                             f'on {1 - train_proportion:.0%} test set')
                predictions = ranker.predict(idxs=test_idxs)
                prec = precision_score(dataset.df.loc[test_idxs, 'label'], predictions > 0.5, zero_division=0)
                rec = recall_score(dataset.df.loc[test_idxs, 'label'], predictions > 0.5, zero_division=0)
                f1 = f1_score(dataset.df.loc[test_idxs, 'label'], predictions > 0.5, zero_division=0)
                logger.debug(f'Recall is {rec:.2%}, precision is {prec:.2%} (F1={f1:.2%})'
                             f'for ranker {ranker.name} on dataset {dataset.KEY} ')

                if best_ranker is None or f1 > best_f1:
                    logger.debug('Found best model so far, remembering and predicting all unseen!')
                    best_rec = rec
                    best_prec = prec
                    best_f1 = f1
                    best_ranker = ranker
                    y_pred = ranker.predict(predict_on_all=False)

            infos.append({
                'params': best_ranker.get_params(preview=False),
                'recall': best_rec,
                'precision': best_prec,
                'f1': best_f1,
                'batch_size': dataset.get_next_batch_size(),
                'batch_i': dataset.last_batch,
            })
            dataset.register_predictions(scores=y_pred,
                                         model=best_ranker.name)
            last_ranker = best_ranker

    return infos
