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
        dataset_repeat: Annotated[int | None, typer.Option(help='if mode==single: the repeat for the dataset')] = None,
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
        tuning_interval: int = 250,
        random_state: int | None = None,
        store_feather: bool = True,
        store_csv: bool = False,
        initial_holdout: int = 0,
        grow_init_batch: bool = True,
        use_fine_tuning: bool = False,
        predict_on_all: bool = False,
        init_nltk: bool = False,
        slurm_gpu: bool = False,
        slurm_hours: int = 20,
        slurm_user: Annotated[str | None, typer.Option(help='email address to notify when done')] = None,
):
    """
    This is the main procedure for pre-computing rankings.

    :param mode_exec:
              DIRECT: Execute the full procedure loop directly
              SLURM: Execute the full procedure loop on SLURM (this generates an sbatch file and submits a job array)
              SINGLE: Execute for specific dataset only
    :param mode_rank: There are two ranking modes:
      * ALL: Compute all scores with all models and all hyperparameters and keep them all
      * BEST: At each batch, determine which model is the best and only use that for ranking unseen
    :param dataset_key: in case `mode_exec` is `SINGLE`, needs to be set, otherwise leave None
    :param num_repeats: Number of times to repeat simulation with different random initialization
    :param min_dataset_size: Reject all datasets with fewer total number of records than this
    :param min_inclusion_rate: Reject all datasets with lower fraction of includes than this
    :param num_random_init: Number of random records to screen before starting prioritisation
    :param grow_init_batch: The random sample might not contain enough includes.
                            If true, will increase starting sample until min_batch_incl found
                            If false, randomly injecting a random relevant document to random sample (not cool)
    :param initial_holdout: Number of relevant documents to hide from initial random sample (target method simulation)


    :param batch_strategy: Select batch strategy (STATIC: every batch has same size, DYNAMIC: varying batch sizes)
    :param stat_batch_size: Size of batches if `batch_strategy` is `STATIC`

            When `batch_strategy` is `DYNAMIC`
            Going to use the largest possible batch size and will grow that maximum over time starting with the lower
            bound. Going to grow the maximum size but not using it all if at least a number of relevant records were
            seen in the current batch.
    :param dyn_min_batch_incl: Minimal number of includes per batch, will retrain if this number was found
                               and len(batch) > min_batch_size OR len(batch) > max_batch_size
    :param dyn_min_batch_size: Minimum (and starting) batch size
    :param dyn_growth_rate: With each batch, going to grow the min_batch_size by this rate
    :param dyn_max_batch_size: Maximum number per batch. This trumps growth rate and min_batch_incl

    :param inject_random_batch_every: If > 0, don't use prioritisation every time but sprinkle in random samples
    :param train_proportion: Which proportion of the screened data to use for training (remainder used for eval)
    :param max_vocab: Maximum vocabulary size (see sklearn.TfidfVectorizer)
    :param max_ngram: Maximum ngram length (see sklearn.TfidfVectorizer, lower bound = 1 fixed)
    :param min_df: Minimum document frequency (see sklearn.TfidfVectorizer)

    :param tuning_interval: Hyperparameter tuning is expensive. Set this to re-use tuning until another
                            set of at least this number of records were screened

    :param random_state: Fix random seed for reproducibility
    :param store_feather: If true, save result as apache arrow file
    :param store_csv: If true, save result as csv file

    :param models: List of models to use (see `it_rankers` or `it_tuning_rankers`); fallback to all if None
    :param use_fine_tuning: Ignored in `BEST` mode, otherwise decides

    :param predict_on_all: If true, will always predict on every document and store all scores
    :param init_nltk: If true, will make sure NLTK datasets are installed

    :param slurm_gpu: If true, will prepare slurm batch script to run on GPU partition
    :param slurm_user: Required for mode `SLURM`: email address to notify when done
    """
    logger.info(f'Data path: {settings.DATA_PATH}')
    settings.ranking_data_path.mkdir(parents=True, exist_ok=True)

    if min_dataset_size <= num_random_init:
        raise ValueError('min_dataset_size must be higher than num_random_init')

    models = assert_models(models)

    if init_nltk:
        prepare_nltk()

    def best_model_ranking_run(dataset: Dataset, repeat: int):
        logger.info(f'Running best model strategy for repeat cycle {repeat}...')

        target_key = f'{dataset.KEY}-{initial_holdout}-{num_random_init}-{repeat}-best'
        logger.info(f'Running ranker {target_key}...')
        logger.debug(f'Checking for {settings.ranking_data_path / f'{target_key}.json'}')
        if (settings.ranking_data_path / f'{target_key}.json').exists():
            logger.info(f' > Skipping {target_key}; simulation already exists')
            return

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
        logger.info('Done with best_model_ranking_run()')

    def rank_using_best(dataset: Dataset):
        logger.info('Rank using best model, initialising dataset...')
        dataset.init(num_random_init=num_random_init, batch_strategy=batch_strategy,
                     stat_batch_size=stat_batch_size, dyn_min_batch_size=dyn_min_batch_size,
                     dyn_max_batch_size=dyn_max_batch_size, inject_random_batch_every=inject_random_batch_every,
                     dyn_min_batch_incl=dyn_min_batch_incl, dyn_growth_rate=dyn_growth_rate,
                     initial_holdout=initial_holdout, grow_init_batch=grow_init_batch,
                     ngram_range=(1, max_ngram), max_features=max_vocab, min_df=min_df)

        if dataset_repeat is not None:
            logger.info(f'Rank using best model, running only repeat {dataset_repeat}')
            best_model_ranking_run(dataset=dataset, repeat=dataset_repeat)
        else:
            logger.info('Rank using best model, start loop...')
            for repeat in range(1, num_repeats + 1):
                best_model_ranking_run(dataset=dataset, repeat=repeat)
                dataset.reset()
        logger.info('Done with rank_using_best()')

    def rank_using_all(dataset: Dataset):
        logger.info('Rank using all, initialising dataset...')
        dataset.init(num_random_init=num_random_init, batch_strategy=batch_strategy,
                     stat_batch_size=stat_batch_size, dyn_min_batch_size=dyn_min_batch_size,
                     dyn_max_batch_size=dyn_max_batch_size, inject_random_batch_every=inject_random_batch_every,
                     dyn_min_batch_incl=dyn_min_batch_incl, dyn_growth_rate=dyn_growth_rate,
                     initial_holdout=initial_holdout, grow_init_batch=grow_init_batch,
                     ngram_range=(1, max_ngram), max_features=max_vocab, min_df=min_df)

        logger.info('Rank using all, start loop...')
        for ranker in it_rankers(models=models, use_fine_tuning=use_fine_tuning):
            for repeat in range(1, num_repeats + 1):
                logger.info(f'Running for repeat cycle {repeat} for ranker {ranker.name}...')
                ranker.attach_dataset(dataset)
                target_key = f'{dataset.KEY}-{initial_holdout}-{repeat}-{ranker.key}'
                logger.info(f'Running ranker {target_key}...')
                logger.debug(f'Checking for {settings.ranking_data_path / f'{target_key}.json'}')
                if (settings.ranking_data_path / f'{target_key}.json').exists():
                    logger.info(f' > Skipping {target_key}; simulation already exists')
                    continue

                ranker.init()

                logger.info('Start loop until no more unseen data...')
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
        logger.info('Done with rank_using_all()')

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
        logger.info(f'Running simulations!')
        for dataset_ in it_filtered_datasets():
            logger.info(f'Running simulations for dataset {dataset_.KEY}!')
            if mode_rank == RankingProcess.BEST:
                logger.debug(f'Running simulations for dataset {dataset_.KEY} using the best model strategy!')
                rank_using_best(dataset=dataset_)
            elif mode_rank == RankingProcess.ALL:
                logger.debug(f'Running simulations for dataset {dataset_.KEY} using all models!')
                rank_using_all(dataset=dataset_)

        logger.info('All done for this job!')

    elif mode_exec == ExecutionMode.SINGLE:
        logger.info(f'Running simulation on dataset: {dataset_key}')
        if dataset_key is None:
            raise AssertionError('Need to set dataset key in single mode')

        dataset_ = read_dataset(key=dataset_key)
        if mode_rank == RankingProcess.BEST:
            logger.debug(f'Running best model simulation on dataset: {dataset_key}')
            rank_using_best(dataset=dataset_)
        elif mode_rank == RankingProcess.ALL:
            logger.debug(f'Running all models simulation on dataset: {dataset_key}')
            rank_using_all(dataset=dataset_)

        logger.info('All done for this job!')

    elif mode_exec == ExecutionMode.SLURM:
        from rankings import TransRanker
        logger.info(f'Preparing slurm script and submitting job!')

        if slurm_user is None:
            raise AssertionError('Must set slurm_user in single mode')

        # Ensure directories are ready
        venv_path = settings.venv_path.absolute().resolve()
        log_path = settings.log_data_path.absolute().resolve()
        log_path.mkdir(parents=True, exist_ok=True)

        # Ensure models are downloaded
        logger.info(f'Making sure all models are available offline!')
        TransRanker.ensure_offline_models()
        prepare_nltk()

        # Prepare some variables to use in the batch file
        datasets = [ds.KEY for ds in it_filtered_datasets()]
        model_args = [f'--models {m}' for m in models]
        rand = f'--random-state {random_state}' if random_state is not None else ''
        sbatch_args = {
            'time': f'{slurm_hours:0>2}:00:00',
            'nodes': '1',
            'mem': '8G',
            'mail-user': f'"{slurm_user}"',
            'output': f'{log_path}/%A_%a.out',
            'error': f'{log_path}/%A_%a.err',
            'chdir': os.getcwd(),
            'array': f'1-{(len(datasets) + 1) * num_repeats}'
        }
        if slurm_gpu:
            sbatch_args |= {
                'gres': 'gpu:1',  # number of GPUs
                'partition': 'gpu',
                'qos': 'gpushort',  # or gpumedium (has MaxJobsPU=None but half priority, see `$ sacctmgr show qos`)
                'cpus-per-task': 5,
            }
        else:
            sbatch_args |= {
                'cpus-per-task': 12,
                'partition': 'standard',
                'qos': 'short',
            }
        sbatch = [f'#SBATCH --{key}={value}' for key, value in sbatch_args.items()]
        # Write slurm batch file
        # For information on array jobs, see: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/array_jobs/
        with open('simulation/rank.slurm', 'w') as slurm_file:
            slurm_file.write(f"""#!/bin/bash

{'\n'.join(sbatch)}
#SBATCH --oversubscribe  # use non-utilized GPUs on busy nodes
#SBATCH --mail-type=END,FAIL  # 'NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'

# Set this to exit the script when an error occurs
set -e
# Set this to print commands before executing
set -o xtrace

# Set up python environment
module load anaconda/2024.10
module load cuda
source "{venv_path}/bin/activate"

# Python env vars
export PYTHONPATH=$PYTHONPATH:{os.getcwd()}
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Using python from $(which python)"
echo "Python version is $(python --version)"

DATASETS=("{'" "'.join(datasets)}")

job=$(($SLURM_ARRAY_TASK_ID - 1))
dataset_idx=$(($job / {num_repeats}))
repeat=$((($job % {num_repeats}) + 1))

echo "array_task_id" $SLURM_ARRAY_TASK_ID " --> job" $job 
echo "dataset" $dataset_idx "repeat" $repeat

python simulation/rank.py SINGLE \\
               --mode-rank {mode_rank.value} \\
               --dataset-key "${{DATASETS[$dataset_idx]}}" \\
               --dataset-repeat "${{repeat}}" \\
               {' '.join(model_args)} \\
               --num-repeats {num_repeats} \\
               --min-dataset-size {min_dataset_size} \\
               --min-inclusion-rate {min_inclusion_rate} \\
               --num-random-init {num_random_init} \\
               --batch-strategy {batch_strategy.value} \\
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
               {rand} \\
               --{'' if store_feather else 'no-'}store-feather \\
               --{'' if store_csv else 'no-'}store-csv \\
               --initial-holdout {initial_holdout} \\
               --{'' if grow_init_batch else 'no-'}grow-init-batch \\
               --{'' if use_fine_tuning else 'no-'}use-fine-tuning \\
               --{'' if predict_on_all else 'no-'}predict-on-all 
""")
        subprocess.run(['sbatch', 'simulation/rank.slurm'])


if __name__ == '__main__':
    app()
