import logging
import re
from typing import Literal, Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger('base')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)


def assess_stepping(labels: Annotated[npt.NDArray[np.int8], Literal[1]]) -> float:
    """Expects a pandas dataframe with the following columns:
     - random (bool): the record was in the random sample run-in phase
     - label (bool): the record was a include/exclude
     - order (int): how many records were seen before this one
    """
    time_to_find = np.arange(len(labels))[labels == True]
    # calculate lags between finds
    time_lags = pd.Series(time_to_find).diff()
    # the rolling average of the previous finds
    avg_preceding = time_lags.rolling(5, center=False).mean().shift()
    # count the proportion where the time lag drops significantly
    return sum(avg_preceding > (time_lags * 4)) / avg_preceding.count()


def assess_gain(labels: Annotated[npt.NDArray[np.int8], Literal[1]]) -> float:
    """
    Expects a pandas dataframe with the following columns:
     - random (bool): the record was in the random sample run-in phase
     - label (bool): the record was a include/exclude
    """
    # calculate the expected slope if screening randomly
    total_records = labels.count()
    total_found = labels.sum()
    slope = total_found / total_records
    # calculate the actual and expected found at each record screened
    n_found = labels.cumsum()
    exp_found = [i * slope for i in range(1, total_records + 1)]
    # sum proportion found at each record screened and normalize by number of records
    return sum((n_found - exp_found) / total_found) / total_records


RE_PARTS = re.compile(r'-(\d+)-(\d)-best')
if __name__ == '__main__':
    BASE_PATH = Path('data/rankings')
    TARGET_PATH = Path('data/plots/curves/')
    TARGET_PATH.mkdir(parents=True, exist_ok=True)

    logger.info('Preparing datasets from stored rankings...')
    DATASETS = {}
    for file in BASE_PATH.glob('*.json'):
        key = RE_PARTS.sub('', file.stem)
        if key not in DATASETS:
            DATASETS[key] = []
        parts = RE_PARTS.findall(file.stem)[0]

        data_file = str(file).replace('.json', '.feather')
        df = pd.read_feather(data_file).sort_values('order')

        DATASETS[key].append({
            'meta': str(file),
            'data': data_file,
            'init_size': int(parts[0]),
            'repeat': int(parts[1]),
            'random': df['random'] == True,
            'labels': df['label'],
            'rand_incl': df[df['random'] == True]['label'].sum(),
            'rand_seen': (df['random'] == True).sum(),
        })

    logger.info(f'Found {len(DATASETS)} datasets')

    colours = ['red', 'green', 'blue']

    stepping_scores = []
    gain_scores = []

    target_recalls = [.8, .85, .9, .95, .98, 1.0]
    worksavings = {f'{tr:.0%}': [] for tr in target_recalls}

    logger.info('Preparing plots and curve niceness scores...')
    for ds, infos in DATASETS.items():
        logger.debug(f'Plotting for {ds} with {len(infos)} repeats...')
        fig, ax = plt.subplots()
        for info in infos:
            n_samples = len(info['labels'])
            n_incl = info['labels'].sum()
            n_cum = info['labels'].cumsum()

            x = np.arange(n_samples)

            gain = assess_gain(info['labels'][~info['random']])
            stepping = assess_stepping(info['labels'][~info['random']])

            gain_scores.append(gain)
            stepping_scores.append(stepping)

            for target_recall in target_recalls:
                idx_fin = x[(n_cum / n_incl) >= target_recall].min()
                worksavings[f'{target_recall:.0%}'].append(1 - (idx_fin / n_samples))

            # plot curve
            ax.plot(x, info['labels'].cumsum(),
                    label=f'#{info['repeat']}: gain={gain:.3f} | stepping={stepping:.3f}',
                    c=colours[info['repeat'] - 1], lw=1)

            # plot vertical line for where random sample ends
            ax.axvline(info['rand_seen'], c=colours[info['repeat'] - 1], ls='--', lw=0.3)

            # plot trajectory for rand_incl/seen
            ax.plot([0, info['rand_seen'], len(info['labels'])],
                    [0, info['rand_incl'], (info['rand_incl'] / info['rand_seen']) * n_samples],
                    '--', c=colours[info['repeat'] - 1], lw=0.5)

            # plot trajectory for perfect ranking
            ax.plot([info['rand_seen'], info['rand_seen'] + n_incl - info['rand_incl']],
                    [info['rand_incl'], n_incl],
                    ':', c=colours[info['repeat'] - 1], lw=0.3)

        ax.set_title(ds)
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(TARGET_PATH / f'{ds}.png')
        plt.close(fig)

    logger.info('Generating histogram summary...')
    BINS = 10
    fig = plt.figure()
    gs0 = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], hspace=0.7)
    ax = fig.add_subplot(gs00[0])
    ax.set_title('Histogram stepping scores')
    pd.Series(stepping_scores).hist(ax=ax, bins=BINS)
    ax = fig.add_subplot(gs00[1])
    ax.set_title('Histogram gain scores')
    pd.Series(gain_scores).hist(ax=ax, bins=BINS)

    gs01 = gridspec.GridSpecFromSubplotSpec(len(target_recalls), 1, subplot_spec=gs0[1], hspace=1)
    pax = None
    for ai, (target_recall, scores) in enumerate(worksavings.items()):
        ax = fig.add_subplot(gs01[ai], sharex=pax, sharey=pax)
        pax = ax
        ax.set_title(f'Histogram left unseen recall target {target_recall}', fontsize=10)
        pd.Series(scores).hist(ax=ax, bins=20, range=(0, 1))
        if ai < (len(worksavings) - 1):
            ax.tick_params(labelbottom=False)

    fig.tight_layout()
    fig.savefig(TARGET_PATH / f'HISTOGRAMS.png')

    logger.info('Finished!')
