import logging
import pandas as pd

from loaders import it_datasets

min_dataset_size = 1000
min_inclusion_rate = 1.
initial_holdout = 0
dyn_min_batch_incl = 2

logger = logging.getLogger('precompute ranks')

infos = []

for _dataset in it_datasets():
    logger.info(f'Checking dataset: {_dataset.KEY}')
    info = {
        'dataset': _dataset.KEY,
        'n_total': _dataset.n_total,
        'n_incl': int(_dataset.n_incl),
        'n_incl_rel': float(_dataset.n_incl / _dataset.n_total * 100),
        'rule_size': _dataset.n_total >= min_dataset_size,
        'rule_incl': (_dataset.n_incl / _dataset.n_total * 100) >= min_inclusion_rate,
        'rule_init': _dataset.n_incl >= (initial_holdout + dyn_min_batch_incl),
    }
    info['rule'] = info['rule_size'] and info['rule_incl'] and info['rule_init']
    infos.append(info)
    logger.debug(info)

    if _dataset.n_total >= min_dataset_size:
        logger.warning(f'SKIP: Dataset {_dataset.KEY} is too small {_dataset.n_total} < {min_dataset_size}')
    elif (_dataset.n_incl / _dataset.n_total) >= min_inclusion_rate:
        logger.warning(f'SKIP: Dataset {_dataset.KEY} inclusion rate too small!')
    elif _dataset.n_incl >= (initial_holdout + dyn_min_batch_incl):
        logger.warning(f'SKIP: Dataset {_dataset.KEY} does not have enough includes for initial holdout!')

df = pd.DataFrame(infos)
df.to_csv('data/dataset_selection.csv', index=False)
