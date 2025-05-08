import pandas as pd
from pathlib import Path
import logging
import sys

"""
This doesnt't yet group by hyperparameter configurations of a stopping method

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_stopping_methods(csv_path):
    """
    Evaluate the performance of different stopping methods.

    Parameters:
    csv_path : str or Path
        Path to the CSV file containing simulation results

    Returns:
    dict
        Dictionary containing evaluation results by dataset, method, recall target, confidence level, and simulation
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # initialise dictionary to store results
    results = {}

    group_cols = ['dataset', 'sim_key', 'sim-rep', 'method', 'method-recall_target', 'method-confidence_level']

    # filter to only columns that actually exist in the dataframe - method-recall_target and method-confidence_level are blank for some methods
    group_cols = [col for col in group_cols if col in df.columns]

    # get unique combinations of dataset, simulation, method, recall target, etc.
    unique_combos = df[group_cols].drop_duplicates()

    logger.info(f"Analysing {len(unique_combos)} unique dataset-method-simulation combinations")

    for _, combo in unique_combos.iterrows():
        dataset = combo['dataset']
        sim_key = combo['sim_key']
        sim_rep = int(combo['sim-rep'])
        method = combo['method']

        # optional parameters
        recall_target = combo.get('method-recall_target', None)
        if recall_target is not None and pd.isna(recall_target):
            recall_target = None

        confidence = combo.get('method-confidence_level', None)
        if confidence is not None and pd.isna(confidence):
            confidence = None

        sim_id = f"sim-{sim_rep}"

        # make conditions for filtering the dataframe to only one combination
        conditions = [
            (df['dataset'] == dataset),
            (df['sim_key'] == sim_key),
            (df['sim-rep'] == sim_rep),
            (df['method'] == method)
        ]

        # only add recall_target condition if it exists
        if recall_target is not None:
            conditions.append(df['method-recall_target'] == recall_target)

        # only add confidence condition if it exists
        if confidence is not None and 'method-confidence_level' in df.columns:
            conditions.append(df['method-confidence_level'] == confidence)

        combo_data = df[pd.concat(conditions, axis=1).all(axis=1)]

        # sort by batch_i for chronological order
        combo_data = combo_data.sort_values('batch_i')

        stop_trigger_col = 'method-safe_to_stop'

        safe_stops = combo_data[combo_data[stop_trigger_col] == True]

        if len(safe_stops) == 0:
            # indicates no safe stopping point found for this combination
            # get the last batch point instead (to represent the method never stopping)
            if len(combo_data) > 0:
                stop_point = combo_data.iloc[-1]  # Use the last batch
                never_stopped = True
            
        else:
            # get the first safe stopping point
            stop_point = safe_stops.iloc[0]
            never_stopped = False

        # calculate achieved recall at stopping point
        n_incl_total = stop_point['n_incl']  # Total number of included studies
        n_incl_seen = stop_point['n_incl_seen']  # Number of included studies seen so far

        # calculate achieved recall
        achieved_recall = n_incl_seen / n_incl_total if n_incl_total > 0 else 1.0

        # get target recall
        if recall_target is not None and not pd.isna(recall_target):
            target_recall = recall_target
        else:
            # default to 0.95 if no target recall for method (i.e., method is target-agnostic)
            target_recall = 0.95

        # calculate cost of overshoot for cases where target was overshot
        overshot = False
        extra_docs_ratio = 0

        if never_stopped:
            overshot = False
            extra_docs_ratio = 0
        elif achieved_recall > target_recall:
            overshot = True

            # find when we first hit the target recall
            # need to sort by n_seen so we find the earliest point where target was met
            combo_data_sorted = combo_data.sort_values('n_seen')

            target_recall_reached = None

            for _, row in combo_data_sorted.iterrows():
                if n_incl_total > 0:
                    current_recall = row['n_incl_seen'] / n_incl_total
                    if current_recall >= target_recall:
                        target_recall_reached = row
                        break

            # in the below, if the “safe-to-stop” value becomes TRUE by the end of exactly the same batch as the one where current_recall first ≥ the target recall (i.e, correspond to the same row in results.csv), stop_point is the same row as recall_target_reached
            # any documents examined inside that final batch are "invisible" to this calculation. 
            # in these cases:
            # extra_docs_ratio is 0 (because the code can only compare whole batches) even if overshot is True (because end-of-batch recall > target),
            if target_recall_reached is not None:
                extra_docs = stop_point['n_seen'] - target_recall_reached['n_seen']
                extra_docs_ratio = extra_docs / stop_point['n_total'] if n_incl_total > 0 else 0

        # store dataset-method-target-confidence results
        if dataset not in results:
            results[dataset] = {}

        if method not in results[dataset]:
            results[dataset][method] = {}

        # create key for the recall target
        target_key = f"target-{target_recall}"
        if target_key not in results[dataset][method]:
            results[dataset][method][target_key] = {}

        # create key for the confidence level
        conf_key = f"conf-{confidence}" if confidence is not None else "conf-default"
        if conf_key not in results[dataset][method][target_key]:
            results[dataset][method][target_key][conf_key] = {}

        # Determine if stopping method indicated "safe-to-stop" in the final batch
        stopped_at_final_batch = False
        if not never_stopped and stop_point["n_seen"] == stop_point["n_total"]:
            stopped_at_final_batch = True

        # Determine if the stopping method indicated "safe-to-stop" in the same batch as target recall was achieved
        stopped_at_target_recall_batch = False
        if overshot and target_recall_reached is not None and stop_point["n_seen"] == target_recall_reached["n_seen"]:
            stopped_at_target_recall_batch = True

        # store results for this specific simulation
        results[dataset][method][target_key][conf_key][sim_id] = {
            'target_recall': target_recall,
            'achieved_recall': achieved_recall,
            'confidence': confidence if confidence is not None else "default",
            'n_seen': stop_point['n_seen'],
            'n_total': stop_point['n_total'],
            'screening_proportion': stop_point['n_seen'] / stop_point['n_total'],
            'overshot': overshot,
            'extra_docs_ratio': extra_docs_ratio,
            'method': method,
            'dataset': dataset,
            'sim_key': sim_key,
            'simulation': sim_rep,
            'ranker': stop_point.get('ranker', None),
            'never_stopped': never_stopped,
            'stopped_at_final_batch': stopped_at_final_batch,
            'stopped_at_target_recall_batch': stopped_at_target_recall_batch,

        }

    return results


def create_effectiveness_csv(results, output_path):
    """
    create a CSV file with stopping method effectiveness results.
    Parameters:
    results : dict
        Dictionary containing evaluation results from evaluate_stopping_methods
    output_path : str or Path
        Path where the CSV should be saved
    """
    rows = []

    # organising by stopping methods
    methods = set()
    for dataset in results:
        for method in results[dataset]:
            methods.add(method)

    for method in sorted(methods):
        # for each method, iterate through all datasets and simulations
        for dataset in sorted(results.keys()):
            if method not in results[dataset]:
                continue

            for target_key in sorted(results[dataset][method].keys()):
                for conf_key in sorted(results[dataset][method][target_key].keys()):
                    for sim_id in sorted(results[dataset][method][target_key][conf_key].keys()):
                        # get result data
                        res = results[dataset][method][target_key][conf_key][sim_id]

                        # create row
                        row = {
                            'stopping_method': method,
                            'dataset': dataset,
                            'simulation': res['simulation'],
                            'sim_key': res['sim_key'],
                            'ranker': res['ranker'],
                            'records_seen': res['n_seen'],
                            'records_total': res['n_total'],
                            'proportion_screened': res['screening_proportion'],
                            'never_stopped': res['never_stopped'],
                            'achieved_recall': res['achieved_recall'],
                            'target_recall': res['target_recall'],
                            'confidence_level': res['confidence'],
                            'overshot_target': res['overshot'],
                            'cost_of_overshoot (prop)': res['extra_docs_ratio'],
                            'stopped_at_final_batch': res['stopped_at_final_batch'],
                            'stopped_at_target_recall_batch': res['stopped_at_target_recall_batch'],
                        }

                        rows.append(row)

    # create dataframe
    if rows:
        df = pd.DataFrame(rows)

        # group by stopping method, then by dataset
        df = df.sort_values(['stopping_method', 'dataset', 'simulation'])

        logger.info(f"Saving effectiveness results to {output_path}")
        df.to_csv(output_path, index=False)

        logger.info(f"Successfully generated stoppingeffectiveness.csv with:")
        logger.info(f"- {len(df['stopping_method'].unique())} stopping methods")
        logger.info(f"- {len(df['dataset'].unique())} datasets")
        logger.info(f"- {len(df)} total evaluations")
    else:
        logger.warning("No results to write to CSV. Is the simulation complete?")


def main():
    base_dir = Path(__file__).parent

    csv_path = base_dir.parent / "data" / "results" / "results.csv"

    output_path = base_dir /"stoppingeffectiveness_notgroupedhyperparametercombination.csv"

    if not csv_path.exists():
        logger.error(f"Results file not found at {csv_path}")
        logger.info("Please run the simulation first using the command:")
        logger.info("python simulation/main.py simulate-stopping --batch-size=100")
        sys.exit(1)

    results = evaluate_stopping_methods(csv_path)

    create_effectiveness_csv(results, output_path)

if __name__ == "__main__":
    main()
