import json
import logging
import os
import subprocess
from hashlib import sha1
from typing import Annotated

import pandas as pd
import typer

from shared.config import settings
from shared.dataset import RankedDataset
from methods import it_methods, get_methods
from shared.util import elapsed_timer

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger('stopping')

app = typer.Typer()


@app.command()
def slurm(
        slurm_user: Annotated[str, typer.Option(help='email address to notify when done')],
        batch_size: int = 100,  # Step size for computing stopping scores
        methods: list[str] | None = None,  # Methods to compute scores for (empty=all)
        results_dir: str = '',
        slurm_hours: int = 8,

) -> None:
    logger.info(f'Preparing slurm script and submitting job!')

    stop_methods = list(get_methods(methods=methods))
    method_keys = [sm.KEY for sm in stop_methods]

    results_path = settings.result_data_path / results_dir
    results_path.mkdir(parents=True, exist_ok=True)

    # Ensure directories are ready
    venv_path = settings.venv_path.absolute().resolve()
    log_path = (settings.log_data_path / 'sim').absolute().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    # Prepare some variables to use in the batch file
    sbatch_args = {
        'time': f'{slurm_hours:0>2}:00:00',
        'nodes': '1',
        'mem': '8G',
        'mail-user': f'"{slurm_user}"',
        'output': f'{log_path}/%A_%a.out',
        'error': f'{log_path}/%A_%a.err',
        'chdir': os.getcwd(),
        'array': f'1-{(len(stop_methods) + 1)}',
        'cpus-per-task': 8,
        'partition': 'standard',
        'qos': 'short',
    }
    sbatch = [f'#SBATCH --{key}={value}' for key, value in sbatch_args.items()]
    # Write slurm batch file
    # For information on array jobs, see: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/array_jobs/
    with open('simulation/simulate.slurm', 'w') as slurm_file:
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

    METHODS=("{'" "'.join(method_keys)}")

    method_idx=$(($SLURM_ARRAY_TASK_ID % {len(method_keys)}))
    python simulation/simulate.py compute-stops \\
                   --batch-size {batch_size} \\
                   --methods "${{METHODS[$method_idx]}}" \\
                   --results-file "{results_path}/simulation-${{METHODS[$method_idx]}}.csv"
    """)
    subprocess.run(['sbatch', 'simulation/simulate.slurm'])


@app.command()
def compute_stops(
        batch_size: int = 100,  # Step size for computing stopping scores
        methods: list[str] | None = None,  # Methods to compute scores for (empty=all)
        results_file: str = 'results.csv',  # CSV Filename for results relative to result data dir
) -> None:
    logger.info(f'Simulating stopping methods with batch size = {batch_size}')
    rows = []

    if (settings.result_data_path / results_file).exists():
        raise FileExistsError(f'For safety, I will not overwrite this file: {settings.result_data_path / results_file}')

    for ranking_info_fp in settings.ranking_data_path.glob('*.json'):
        dataset = RankedDataset(ranking_info_fp=ranking_info_fp)

        ranking_fp = f'{ranking_info_fp.with_suffix('')}.feather'

        logger.info(f'Ranking from: {ranking_fp}')
        logger.debug(f'Info from {ranking_info_fp}')

        for batch_i, (batches, labels, scores, is_prioritised) in enumerate(
                dataset.it_cum_batches(batch_size=batch_size)
        ):
            logger.info(f'Running batch {batch_i} ({len(batches):,}/{dataset.n_total:,})')
            base_entry = {
                'dataset': dataset.dataset,
                'sim-rep': dataset.repeat,
                'sim_key': dataset.info['key'],
                'batch_i': batch_i,
                'n_total': dataset.n_total,
                'n_seen': len(labels),
                'n_unseen': dataset.n_total - len(labels),
                'n_incl': dataset.n_incl,
                'n_incl_seen': labels.sum(),
                'n_incl_batch': labels[-batch_size:].sum(),
                'n_records_batch': batch_size,
                **{
                    f'ranker-{k}': v
                    for k, v in dataset.info.items()
                    if k.startswith('model-')
                },
                **{
                    f'sampling-{k}': v
                    for k, v in dataset.info.items()
                    if k.startswith('batch-')
                }
            }

            for method in it_methods(dataset=dataset, methods=methods):
                with elapsed_timer(logger, f'Evaluating method {method.KEY}'):
                    for paramset in method.parameter_options():
                        stop_result = method.compute(
                            list_of_labels=labels,
                            list_of_model_scores=scores,
                            is_prioritised=is_prioritised,
                            **paramset)

                        rows.append({
                            **base_entry,
                            'method': method.KEY,
                            'safe_to_stop': stop_result.safe_to_stop,
                            'method-hash': (method.KEY + '-' +
                                            sha1(json.dumps(paramset, sort_keys=True).encode('utf-8')).hexdigest()),
                            **{
                                f'method-{k}': v
                                for k, v in stop_result.model_dump().items()
                            },
                        })

        # Write entire log to disk after every dataset
        pd.DataFrame(rows).to_csv(settings.result_data_path / results_file, index=False)


if __name__ == '__main__':
    app()
