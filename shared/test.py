import logging
from typing import Type

import numpy as np

from shared.dataset import RankedDataset
from shared.method import Method, _MethodParams, _LogEntry
from shared.util import elapsed_timer
from matplotlib import pyplot as plt


def test_method(Method: Type[Method],
                paramset: _MethodParams,
                dataset_i: int = 0,
                batch_size: int = 25,
                loglevel: int | str = logging.DEBUG) -> tuple[RankedDataset, list[_LogEntry]]:
    from shared.config import settings
    from shared.dataset import RankedDataset
    import logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=loglevel)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger('test')

    fp = list(settings.ranking_data_path.glob('*.json'))[dataset_i]
    dataset = RankedDataset(ranking_info_fp=fp)
    logger.info(f'Ranking from: {fp}')
    logger.info(dataset)

    results = []
    for batch_i, (_, bounds, (labels, scores, is_prioritised)) in enumerate(dataset.cum_batches(batch_size=batch_size)):
        base_entry = {
            'batch_i': batch_i,
            'n_total': dataset.n_total,
            'n_seen': len(labels),
            'n_unseen': dataset.n_total - len(labels),
            'n_incl': dataset.n_incl,
            'n_incl_seen': labels.sum(),
            'n_incl_batch': labels[-batch_size:].sum(),
            'n_records_batch': batch_size,
        }
        with elapsed_timer(logger, f'Batch {batch_i} took {Method.KEY}'):
            results.append(base_entry | Method.compute(
                n_total = dataset.n_total,
                labels=labels,
                scores=scores,
                full_labels=dataset.labels,
                bounds=bounds,
                is_prioritised=is_prioritised,
                **paramset))
            logger.debug(f'  -> {results[-1]}')
    return dataset, results


def plots(dataset: RankedDataset, results: list[_LogEntry], params: _MethodParams) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(layout='constrained', dpi=150, figsize=(12, 10))
    labels = dataset.ranking['label']
    rand = dataset.ranking.reset_index(drop=True)['random'] == True
    curve = labels.reset_index(drop=True).cumsum()
    curve_rand = curve[rand]
    curve_prio = curve[~rand]

    n_docs = dataset.n_total
    rel_curve = curve / dataset.n_incl

    ax.axvline(np.argwhere(rel_curve >= 0.8).min(), label='Target 80%', ls='--', alpha=0.4, lw=0.5)
    ax.axvline(np.argwhere(rel_curve >= 0.9).min(), label='Target 90%', ls='--', alpha=0.4, lw=0.5)
    ax.axvline(np.argwhere(rel_curve >= 0.95).min(), label='Target 95%', ls='--', alpha=0.4, lw=0.5)
    ax.axvline(np.argwhere(rel_curve >= 1).min(), label='Target 100%', ls='--', alpha=0.4, lw=0.5)
    ax.axhline(curve.max(), ls='--', alpha=0.4, lw=0.5)

    # vertical line for where random sample ends
    ax.axvline(len(curve_rand), ls='--', lw=0.3, label=f'n_rand_init={len(curve_rand):,}')
    # plot trajectory for rand_incl/seen
    ax.plot([0, len(curve_rand), n_docs], [0, curve_rand.max(), (curve_rand.max() / len(curve_rand)) * n_docs],
            ls='--', lw=0.5, label='Random inclusion rate')

    # plot trajectory for perfect ranking
    ax.plot([len(curve_rand), len(curve_rand) + curve.max() - curve_rand.max()], [curve_rand.max(), curve.max()],
            ls='--', lw=0.5, label='Perfect prioritisation')
    # area random init
    ax.fill_between(curve.index, 0, curve.max(), where=curve.index < len(curve_rand), facecolor='grey', alpha=0.1)

    offset = 0
    for _, l in dataset.ranking.groupby('batch'):
        offset += len(l)
        ax.vlines(offset, 0, 5, lw=1, color='black')

    offset = 0
    for res in results:
        ax.fill_between(np.arange(offset, res['n_seen']), 0, 5,
                        facecolor='green' if res['safe_to_stop'] else 'red', alpha=0.2)
        offset = res['n_seen']

    ax.plot(curve_rand, lw=2, color='grey', label='Inclusions (random)')
    ax.plot(curve_prio, lw=2, color='black', label='Inclusions (prioritised)')

    ax.set_ylim(0, curve.max() + 5)
    ax.set_xlim(0, n_docs)
    ax.set_title(f'{dataset.info['key']}', fontsize=10)
    ax.set_xlabel(f'Number of seen documents (total={dataset.n_total:,})')
    ax.set_ylabel(f'Number of included documents (total={curve.max():,})')
    fig.legend(
        loc='outside right center',
        title='\n'.join([f'{k}: {v}' for k, v in params.items()]),
    )

    return fig, ax
