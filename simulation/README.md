# Stopping method benchmark

In the following examples, we assume that you've already set up and activated an environment and that you are in the
root directory of this repository

```bash
# Change to repository root
cd /path/to/stopping-methods
# Create virtual environment (python 3.11 or 3.12 may also work)
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preparing datasets

There are several generic readers to use.

* CSV files in `data/generic-csv`, each file will be considered a dataset and should contain the columns
  `title, abstract, label_abs`
* jsonl files in `data/generic-jsonl`, each file will be considered a dataset, each row should be a valid dump of
  `Record` as defined in `shared.dataset`
* pairs of RIS files in `data/generic-paired-ris`, should have matching filenames, one ending in `_include.ris`, the
  other in `_exclude.ris`. The framework will read and parse these files and automatically assign respective abstract
  include labels

Some other existing datasets are fetched on first use via

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/stopping-methods && python simulation/main.py prepare-datasets
```

This will, for example, download the synergy dataset and store it locally and maybe fetch additional information to
enrich the dataset from openalex.

### HPC setup
```bash
cd /p/tmp/user/
git clone git@github.com:destiny-evidence/stopping-methods.git
cd stopping-methods
module load anaconda/2024.10
# verify we are really using the correct python
which python
python --version
# set up virtualenv
python -m venv data/venv
source data/venv/bin/activate
pip install -r requirements.txt

# pre-compute rankings
 PYTHONPATH=. python simulation/rank.py SLURM --models trans-rank --models svm --models lightgbm --models sgd  --models logreg \
                                --dyn-min-batch-size 25 --dyn-max-batch-size 200 --dyn-min-batch-incl 2 \
                                --num-random-init 500 --min-dataset-size 1000 --num-repeats 3 \
                                --min-inclusion-rate 0.01 --tuning-interval 4 --store-feather --slurm-user "???@pik-potsdam.de" --slurm-hours 23 --slurm-gpu
# or
 PYTHONPATH=. python simulation/rank.py SLURM --models trans-rank --models svm --models lightgbm --models sgd  --models logreg \
                                --dyn-min-batch-size 25 --dyn-max-batch-size 200 --dyn-min-batch-incl 2 \
                                --num-random-init 500 --min-dataset-size 1000 --num-repeats 3 \
                                --min-inclusion-rate 0.01 --tuning-interval 4 --store-feather --slurm-user "???@pik-potsdam.de" --slurm-hours 23
```

## Pre-computing rankings

The following will pre-compute all rankings that are now yet there. You can run this script repeatedly and it will
only perform runs that it doesn't have cached data for yet.
In general, it will create a ranking for all datasets, for all models, for all respective configurations

```bash
# Recommended env vars (assuming you already downloaded necessary huggingface models)
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# get help 
export PYTHONPATH=$PYTHONPATH:/path/to/stopping-methods && python simulation/main.py precompute-rankings --help
# run
export PYTHONPATH=$PYTHONPATH:/path/to/stopping-methods && python simulation/main.py precompute-rankings --dyn-max-batch-size=2000 --use-svm --use-reg --use-sdg --use-fine-tuning
```

Options

* `use_svm / use_sdg / use_reg` set to true which ranking models to use. Respective setups are specified in
  `it_rankers()` of `rank.py`
* `use_fine_tuning` hyperparameter tuning is a little slow, hence you can turn it on or off
* `num_repeats: int = 3` number of times to repeat each training with each dataset/model/parameter combination
* `inject_random_batch_every: int = 0` if >1, this will skip training/prediction every n-th batch and instead use a
  random sample of unseen data
* `predict_on_all: bool = True` if True, will always compute prediction scores for all records in the dataset (instead
  of only for unseen ones)

### Static batch size

Set `--batch-strategy STATIC` and choose a fixed batch size via `--stat-batch-size 100`.
Note, that this batch size is only for the ranking, stopping criteria are evaluated with a different batch size.

### Dynamic batch size

It does make sense to dynamically adjust the batch size, namely to grow it after each iteration.
The growth rate is specified via `dyn_growth_rate`, so the next batch size will always be
`target=len(last_batch) * dyn_growth_rate`.
However, it does not really make sense to retrain the model if you haven't seen any new included examples.
Via `dyn_min_batch_incl` you can set how many labelled includes should be in a batch at least.
The batch will be extended from `target` until we find the respective number of includes.
We may also keep the batch smaller if we found includes earlier.
To keep this in check, we have a minimum batch size (`dyn_min_batch_size`) and a maximum batch size (
`dyn_max_batch_size`).

A sensible setup could be:

```
dyn_min_batch_incl = 5
dyn_min_batch_size = 100
dyn_growth_rate = 0.1
dyn_max_batch_size = 2000
```

Note, that this batch size is only for the ranking, stopping criteria are evaluated with a different batch size.

## Compute stopping

The following will compute all the scores for all the pre-computed rankings.

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/stopping-methods && python simulation/main.py simulate-stopping --batch-size=100 --results_file results.csv


# via slurm
export PYTHONPATH=$PYTHONPATH:. && python simulation/simulate.py slurm --slurm-user=name@pik-potsdam.de --batch-size=15 --slurm-hours=12
# check status
squeue --me -t all
# clearing
rm  data/results/simulation-*
```

Options:

* `methods` list of methods to use (see respective class variables `KEY`); if empty, use all

In general, each stopping method has a set of parameters that it will cycle through (see `parameter_options()` class
methods).
For each batch in a dataset, scores are computed for all methods and combinations of parameters.
All this is written to a large csv file in the `data/results/` folder

### Structure of results.csv

```
dataset  <- KEY of the dataset
ranker   <- Ranking model classname
sim-rep  <- number of the ranking run
sim_key  <- unique hash for these ranking hyperparameters
batch_i  <- batch number for the stopping simulation

n_total, n_incl                <- global dataset statistic (total size and number of includes)
n_seen, n_unseen, n_incl_seen  <- progress statistics (how many seen so far, and how many of those were includes)
n_incl_batch, n_records_batch  <- size of this batch and number of includes in this batch

method, method-KEY  <- key of the respective stopping classmethod KEY
method-hash         <- unique hash for this hyperparameter setting
safe_to_stop        <- True when criterion triggered in this batch (might be false later on again, so look for first occurrence of this being true)

method-*            <- outputs and hyperparameters of stopping method
ranker-model-*      <- hyperparameters of the ranking model; not all fields filled (depends on respective model)
sampling-batch-*    <- hyperparameters of the batch sizing during ranking
```

This is a very verbose and complete data structure which is also intimidating.
The following lets you iterate over data from individual runs and methods

```python
for (hash_ranker, hash_method, repeat), sub_df in df.groupby(['sim_key', 'method-hash', 'sim-rep']):
    simulation = sub_df.sort_values(by=['batch_i'])
    info = simulation.iloc[0]
    print(
        f'Dataset "{info['dataset']}" ranked by "{info['ranker']}" stopped by "{info['method']}" (repeat {repeat} via {hash_method} / {hash_ranker})')
    for _, step in simulation.iterrows():
        recall = step['n_incl_seen'] / step['n_incl']

        print(f'Batch {step['batch_i']}: {step['n_seen']:,}/{step['n_total']:,} seen; '
              f'{step['n_incl_seen']:,}/{step['n_incl']:,} includes found; '
              f'recall={recall:.2%} | safe to stop: {step['safe_to_stop']}')

    print('---')
```