import pandas as pd
from pathlib import Path
import logging
import sys
import json

"""
This code groups by hyperparameter configurations of a stopping method. 
If target recall not specified, defaults to a value of 0.95 for overshoot calculation (Line 145)

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
        Dictionary containing evaluation results by dataset, method, recall target, confidence level, hyperparameters, and simulation
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # initialise dictionary to store results
    results = {}

    # Base grouping columns
    base_group_cols = ['dataset', 'sim_key', 'sim-rep', 'method']
    
    # Filter to only columns that actually exist in the dataframe
    base_group_cols = [col for col in base_group_cols if col in df.columns]

    # Get unique method names to collect method-specific hyperparameters
    methods = df['method'].unique()
    
    # Process each method separately to identify its hyperparameters
    for method_name in methods:
        logger.info(f"Processing method: {method_name}")
        
        # Filter dataframe to only current method
        method_df = df[df['method'] == method_name]
        
        # Identify method- hyperparameters which always have a value (columns starting with "method-")
        common_method_cols = ['method-KEY', 'method-safe_to_stop', 'method-hash']
        
        # Identify method-target_recall and method-confidence_level (important)
        special_params = ['method-recall_target', 'method-confidence_level']
        special_params_available = [col for col in special_params if col in method_df.columns]
        
        # Get all method-specific columns
        method_cols = [col for col in method_df.columns if col.startswith('method-') 
                      and col not in common_method_cols 
                      and col not in special_params
                      and not pd.isna(method_df[col]).all()]  # Exclude columns that are all NaN
        
        # Group by base columns, special params, and method-specific hyperparameters
        method_group_cols = base_group_cols + special_params_available + method_cols
        
        logger.info(f"Method {method_name} has {len(method_cols)} hyperparameters: {', '.join(method_cols)}")
        
        # Get unique combinations for method
        method_unique_combos = method_df[method_group_cols].drop_duplicates()
        
        logger.info(f"Method {method_name} has {len(method_unique_combos)} unique parameter combinations")
        
        # Iterate through each unique combination
        for _, combo in method_unique_combos.iterrows():
            dataset = combo['dataset']
            sim_key = combo['sim_key']
            sim_rep = int(combo['sim-rep'])
            method = combo['method']

            # Special parameters (if available)
            recall_target = combo.get('method-recall_target', None)
            if recall_target is not None and pd.isna(recall_target):
                recall_target = None

            confidence = combo.get('method-confidence_level', None)
            if confidence is not None and pd.isna(confidence):
                confidence = None

            # Create conditions for filtering dataset for these combinations
            conditions = [
                (df['dataset'] == dataset),
                (df['sim_key'] == sim_key),
                (df['sim-rep'] == sim_rep),
                (df['method'] == method)
            ]

            # Add method-target_recall and method-confidence_level parameters if not None
            if recall_target is not None:
                conditions.append(df['method-recall_target'] == recall_target)

            if confidence is not None and 'method-confidence_level' in df.columns:
                conditions.append(df['method-confidence_level'] == confidence)
            
            # Add other method-specific hyperparameter conditions
            for param in method_cols:
                param_value = combo.get(param, None)
                if param_value is not None and not pd.isna(param_value):
                    conditions.append(df[param] == param_value)

            # Get data for this specific combination
            combo_data = df[pd.concat(conditions, axis=1).all(axis=1)]

            # Sort by batch_i (chronological order)
            combo_data = combo_data.sort_values('batch_i')

            stop_trigger_col = 'method-safe_to_stop'

            safe_stops = combo_data[combo_data[stop_trigger_col] == True]

            if len(safe_stops) == 0:
                # Indicates no safe stopping point found for this combination
                # Get the last batch point instead (to represent the method never stopping)
                stop_point = combo_data.iloc[-1]  # last batch
                never_stopped = True
                
            else:
                # Get the first safe stopping point
                stop_point = safe_stops.iloc[0]
                never_stopped = False

            # Calculate achieved recall at stopping point
            n_incl_total = stop_point['n_incl']  # Total number of included studies
            n_incl_seen = stop_point['n_incl_seen']  # Number of included studies seen so far

            # Calculate achieved recall
            achieved_recall = n_incl_seen / n_incl_total if n_incl_total > 0 else 1.0

            # Get target recall
            if recall_target is not None and not pd.isna(recall_target):
                target_recall = recall_target
            else:
                # Default to 0.95 if no target recall for method (i.e., method is target-agnostic)
                target_recall = 0.95

            # Calculate cost of overshoot (proportion of dataset screened) for cases where target was overshot
            overshot = False
            extra_docs_ratio = 0

            if never_stopped:
                overshot = False
                extra_docs_ratio = 0
            elif achieved_recall > target_recall:
                overshot = True

                # Find when we first hit the target recall
                # Need to sort by n_seen so we find the earliest point where target was met
                combo_data_sorted_recall_cacl = combo_data.sort_values('n_seen')

                target_recall_reached = None

                for _, row in combo_data_sorted_recall_cacl.iterrows():
                    if n_incl_total > 0:
                        current_recall = row['n_incl_seen'] / n_incl_total
                        if current_recall >= target_recall:
                            target_recall_reached = row
                            break

                # Calculate extra docs ratio
                if target_recall_reached is not None:
                    extra_docs = stop_point['n_seen'] - target_recall_reached['n_seen']
                    extra_docs_ratio = extra_docs / stop_point['n_total'] if n_incl_total > 0 else 0

            # Create unique hyperparameter key
            hyperparam_values = {}
            for param in method_cols:
                param_value = combo.get(param, None)
                # Convert list-like strings to
                #  strings for better readability
                if isinstance(param_value, str) and param_value.startswith('[') and param_value.endswith(']'):
                    try:
                        # Try to parse as JSON if it looks like a list or dict
                        param_value = json.dumps(json.loads(param_value))
                    except:
                        pass
                hyperparam_values[param] = param_value
            
            # Create a unique hyperparameter key
            param_key = ";".join([f"{param.replace('method-', '')}={str(value)}" 
                                for param, value in sorted(hyperparam_values.items())])
            
            if not param_key:  # If no hyperparameters, use a default key
                param_key = "no hyperparameters"
            
            # Create simulation ID
            sim_id = f"sim-{sim_rep}"

            # Store dataset in results
            if dataset not in results:
                results[dataset] = {}

            # Store method in results
            if method not in results[dataset]:
                results[dataset][method] = {}

            # Create key for the recall target
            target_key = f"target-{target_recall}"
            if target_key not in results[dataset][method]:
                results[dataset][method][target_key] = {}

            # Create key for the confidence level
            conf_key = f"conf-{confidence}" if confidence is not None else "conf-default"
            if conf_key not in results[dataset][method][target_key]:
                results[dataset][method][target_key][conf_key] = {}
                
            # Create key for the hyperparameters
            if param_key not in results[dataset][method][target_key][conf_key]:
                results[dataset][method][target_key][conf_key][param_key] = {}

            # Determine if stopping method indicated "safe-to-stop" in the final batch
            stopped_at_final_batch = False
            if not never_stopped and stop_point["n_seen"] == stop_point["n_total"]:
                stopped_at_final_batch = True

            # Determine if the stopping method indicated "safe-to-stop" in the same batch as target recall was achieved
            stopped_at_target_recall_batch = False
            if overshot and target_recall_reached is not None and stop_point["n_seen"] == target_recall_reached["n_seen"]:
                stopped_at_target_recall_batch = True

            # Store results for this specific simulation
            results[dataset][method][target_key][conf_key][param_key][sim_id] = {
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
                'hyperparameters': hyperparam_values,
            }

    return results


def create_effectiveness_csv(results, output_path):
    """
    Create a CSV file with stopping method effectiveness results.
    Parameters:
    results : dict
        Dictionary containing evaluation results from evaluate_stopping_methods
    output_path : str or Path
        Path where the CSV should be saved
    """
    rows = []

    # Organizing by stopping methods
    methods = set()
    for dataset in results:
        for method in results[dataset]:
            methods.add(method)

    for method in sorted(methods):
        # For each method, iterate through all datasets and simulations
        for dataset in sorted(results.keys()):
            if method not in results[dataset]:
                continue

            for target_key in sorted(results[dataset][method].keys()):
                for conf_key in sorted(results[dataset][method][target_key].keys()):
                    for param_key in sorted(results[dataset][method][target_key][conf_key].keys()):
                        for sim_id in sorted(results[dataset][method][target_key][conf_key][param_key].keys()):
                            # Get result data
                            res = results[dataset][method][target_key][conf_key][param_key][sim_id]

                            # Create row
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
                                'hyperparameter_key': param_key,
                                'overshot_target': res['overshot'],
                                'cost_of_overshoot (prop)': res['extra_docs_ratio'],
                                'stopped_at_final_batch': res['stopped_at_final_batch'],
                                'stopped_at_target_recall_batch': res['stopped_at_target_recall_batch'],
                            }
                            
                            # Add hyperparameters as separate columns
                            for param, value in res['hyperparameters'].items():
                                # Extract parameter name without "method-" prefix
                                param_name = param.replace('method-', '')
                                row[param_name] = value

                            rows.append(row)

    # Create dataframe
    if rows:
        df = pd.DataFrame(rows)

        # Group by stopping method, hyperparameters, then by dataset
        sort_cols = ['stopping_method', 'hyperparameter_key', 'dataset', 'simulation']
        sort_cols = [col for col in sort_cols if col in df.columns]
        df = df.sort_values(sort_cols)

        logger.info(f"Saving effectiveness results to {output_path}")
        df.to_csv(output_path, index=False)

        logger.info(f"Successfully generated stoppingeffectiveness.csv with:")
        logger.info(f"- {len(df['stopping_method'].unique())} stopping methods")
        logger.info(f"- {len(df['dataset'].unique())} datasets")
        unique_params = df.groupby(['stopping_method', 'hyperparameter_key']).size().reset_index()[['stopping_method', 'hyperparameter_key']]
        logger.info(f"- {len(unique_params)} unique stopping method configurations")
        logger.info(f"- {len(df)} total evaluations")
    else:
        logger.warning("No results to write to CSV. Is the simulation complete?")


def main():
    base_dir = Path(__file__).parent

    csv_path = base_dir.parent / "data" / "results" / "results.csv"

    output_path = base_dir / "stoppingeffectiveness.csv"

    if not csv_path.exists():
        logger.error(f"Results file not found at {csv_path}")
        logger.info("Please run the simulation first using the command:")
        logger.info("python simulation/main.py simulate-stopping --batch-size=100")
        sys.exit(1)

    results = evaluate_stopping_methods(csv_path)

    create_effectiveness_csv(results, output_path)

if __name__ == "__main__":
    main()