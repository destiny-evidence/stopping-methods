import logging

import typer
from datasets import prepare_collections
from simulation.rank import produce_rankings, best_model_ranking
from simulation.simulate import compute_stops

logger = logging.getLogger('simulation')

app = typer.Typer()
# app.add_typer(rank_app, name='precompute-rankings', help='Compute rankings for all datasets using all rankers')
# app.add_typer(simulate_app, name='simulate-stopping', help='Apply all stopping methods to all pre-computed datasets')
app.command(
    name='precompute-rankings',
    help='Compute rankings for all datasets using all rankers')(produce_rankings)
app.command(
    name='precompute-bm-rankings',
    help='Compute rankings for all datasets using best ranker at each batch')(best_model_ranking)
app.command(
    name='simulate-stopping',
    help='Apply all stopping methods to all pre-computed datasets')(compute_stops)
app.command(
    name='prepare-datasets',
    help='Download datasets and collections')(prepare_collections)

if __name__ == '__main__':
    app()
