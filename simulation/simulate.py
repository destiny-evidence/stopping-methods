import json
import logging
from hashlib import sha1

import pandas as pd
import typer

from shared.config import settings
from shared.dataset import RankedDataset
from methods import it_methods
from shared.util import elapsed_timer

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger('stopping')

app = typer.Typer()


@app.command()
def compute_stops(
        batch_size: int = 100,  # Step size for computing stopping scores
        methods: list[str] | None = None,  # Methods to compute scores for (empty=all)
        results_file: str = 'results.csv',  # CSV Filename for results relative to result data dir
) -> None:
    logger.info(f'Simulating stopping methods with batch size = {batch_size}')
    rows = []

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
