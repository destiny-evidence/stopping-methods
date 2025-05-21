import logging
import os
import subprocess
from enum import Enum
from typing import Annotated, Generator

import typer
from rankings import assert_models, it_rankers
from rankings.use_best import best_model_ranking as bm_ranking
from shared.config import settings
from shared.dataset import BatchStrategy, Dataset
from loaders import it_datasets, read_dataset
from shared.disk import json_dumps

logger = logging.getLogger('precompute ranks')

app = typer.Typer()


class RankingProcess(str, Enum):
    ALL = 'ALL'
    BEST = 'BEST'


class ExecutionMode(str, Enum):
    SLURM = 'SLURM'
    DIRECT = 'DIRECT'
    SINGLE = 'SINGLE'


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
        mode_exec: ExecutionMode,
        mode_rank: RankingProcess = RankingProcess.BEST,
        dataset_key: Annotated[str | None, typer.Option(help='if mode==single: the key for the dataset')] = None,
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
        initial_holdout: int = 0,
        grow_init_batch: bool = True,
        use_fine_tuning: bool = False,
        predict_on_all: bool = True,
        init_nltk: bool = False,
        slurm_user: Annotated[str | None, typer.Option(help='email address to notify when done')] = None,
):
    logger.info(f'Data path: {settings.DATA_PATH}')
    settings.ranking_data_path.mkdir(parents=True, exist_ok=True)

    if min_dataset_size <= num_random_init:
        raise ValueError('min_dataset_size must be higher than num_random_init')

    models = assert_models(models)

    if init_nltk:
        prepare_nltk()

    def rank_using_best(dataset: Dataset):
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
                             initial_holdout=initial_holdout, grow_init_batch=grow_init_batch,
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

    def rank_using_all(dataset: Dataset):
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

    def it_filtered_datasets() -> Generator[Dataset, None, None]:
        for _dataset in it_datasets():
            logger.info(f'Running simulation on dataset: {_dataset.KEY}')
            logger.info(f'  n_incl={_dataset.n_incl}, n_total={_dataset.n_total} '
                        f'=> {_dataset.n_incl / _dataset.n_total:.2%}')

            if _dataset.n_total < min_dataset_size:
                logger.warning(f'SKIP: Dataset {_dataset.KEY} is too small {_dataset.n_total} < {min_dataset_size}')
                continue
            if (_dataset.n_incl / _dataset.n_total) < min_inclusion_rate:
                logger.warning(f'SKIP: Dataset {_dataset.KEY} inclusion rate too small!')
                continue
            if _dataset.n_incl < (initial_holdout + dyn_min_batch_incl):
                logger.warning(f'SKIP: Dataset {_dataset.KEY} does not have enough includes for initial holdout!')
                continue

            yield _dataset

    if mode_exec == ExecutionMode.DIRECT:
        for dataset_ in it_filtered_datasets():
            if mode_rank == RankingProcess.BEST:
                rank_using_best(dataset=dataset_)
            elif mode_rank == RankingProcess.ALL:
                rank_using_all(dataset=dataset_)

    elif mode_exec == ExecutionMode.SINGLE:
        if dataset_key is None:
            raise AssertionError('Need to set dataset key in single mode')
        dataset_ = read_dataset(key=dataset_key)
        if mode_rank == RankingProcess.BEST:
            rank_using_best(dataset=dataset_)
        elif mode_rank == RankingProcess.ALL:
            rank_using_all(dataset=dataset_)

    elif mode_exec == ExecutionMode.SLURM:
        venv_path = settings.venv_path.absolute().resolve()
        log_path = settings.log_data_path.absolute().resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        if slurm_user is None:
            raise AssertionError('Must set slurm_user in single mode')

        # datasets = [ds.KEY for ds in it_filtered_datasets()]
        datasets = ['a', 'b', 'c']
        model_args = [f'--models {m}' for m in models]
        rand = f'--random-state {random_state} \\' if random_state is not None else ''

        with open('simulation/rank.slurm', 'w') as slurm_file:
            slurm_file.write(f"""#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --qos=gpushort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # number of GPUs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --oversubscribe  # use non-utilized GPUs on busy nodes
#SBATCH --mail-type=END,FAIL  # 'NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'
#SBATCH --output={log_path}/%A_%a.out
#SBATCH --error={log_path}/%A_%a.err
#SBATCH --chdir={os.getcwd()}
#SBATCH --mail-user={slurm_user}
#SBATCH --array=1-{len(datasets)}

# Set this to exit the script when an error occurs
set -e
# Set this to print commands before executing
set -o xtrace

# Set up python environment
module load anaconda/2024.10
module load cuda
source "{venv_path}/bin/activate"

# Python env vars
export PYTHONPATH=$PYTHONPATH:$DIR_CODE
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Using python from $(which python)"
echo "Python version is $(python --version)"

DATASETS=("{'" "'.join(datasets)}")

python rank.py SINGLE \\
               --mode_rank {mode_rank} \\
               --dataset-key ${{DATASETS[$SLURM_ARRAY_TASK_ID]}} \\
               {model_args} \\
               --num-repeats {num_repeats} \\
               --min-dataset-size {min_dataset_size} \\
               --min-inclusion-rate {min_inclusion_rate} \\
               --num-random-init {num_random_init} \\
               --batch-strategy {batch_strategy} \\
               --stat-batch-size {stat_batch_size} \\
               --dyn-min-batch-incl {dyn_min_batch_incl} \\
               --dyn-min-batch-size {dyn_min_batch_size} \\
               --dyn-growth-rate {dyn_growth_rate} \\
               --dyn-max-batch-size {dyn_max_batch_size} \\
               --inject-random-batch-every {inject_random_batch_every} \\
               --train-proportion {train_proportion} \\
               --max-vocab {max_vocab} \\
               --max-ngram {max_ngram} \\
               --min-df {min_df} \\
               --tuning-interval {tuning_interval} \\
               {rand}
               --{'' if store_feather else 'no-'}store-feather \\
               --{'' if store_csv else 'no-'}store-csv \\
               --initial-holdout {initial_holdout} \\
               --{'' if grow_init_batch else 'no-'}grow-init-batch \\
               --{'' if use_fine_tuning else 'no-'}use-fine-tuning \\
               --{'' if predict_on_all else 'no-'}predict_on_all 
""")
            subprocess.run(['sbatch', 'simulation/rank.slurm'])


if __name__ == '__main__':
    app()
