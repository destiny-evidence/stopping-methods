# Simulation evaluations

We have millions of logs from the simulations. The following snippet may help to understand how to correctly iterate
the data to evaluate a specific simulation (for a dataset, method, and parameter setting):
```python
for (hash_ranker, hash_method, repeat), sub_df in df.groupby(['sim_key', 'method-hash', 'sim-rep']):
    simulation = sub_df.sort_values(by=['batch_i'])
    info = simulation.iloc[0]
    print(f'Dataset "{info['dataset']}" ranked by "{info['ranker']}" stopped by "{info['method']}" (repeat {repeat} via {hash_method} / {hash_ranker})')
    for _, step in simulation.iterrows():
        recall = step['n_incl_seen'] / step['n_incl']

        print(f'Batch {step['batch_i']}: {step['n_seen']:,}/{step['n_total']:,} seen; '
              f'{step['n_incl_seen']:,}/{step['n_incl']:,} includes found; '
              f'recall={recall:.2%} | safe to stop: {step['safe_to_stop']}')

    print('---')
```

* `curve_quality.py`: Produces some experimental scores to quantify how well the prioritisation works and the inclusion curve looks like
* `dataset_selection.py`: Iterates all datasets we have in store and produces a summary table which datasets are actually used (we exclude small and very low inclusion rate datasets)
* `MultiCurvePlot.ipynb`: Produces a composite plot of all pre-computed rankings and some other related plots
* `ExploreResults.ipynb`: Main evaluation