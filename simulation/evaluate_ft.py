import sys
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_stopping_methods(csv_path):
    """
    Evaluate the performance of different systematic review stopping methods:
    - Calculate the actual recall achieved when methods indicate it's safe to stop
    - Compare against target recall
    - Calculate screening efficiency for methods that overshot their targets
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file containing the systematic review results
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results by dataset, method, recall target, confidence level, and simulation
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Dictionary to store results
    results = {}
    
    group_cols = ['dataset', 'sim_key', 'sim-rep', 'method', 'method-recall_target', 'method-confidence_level']
    unique_combos = df[group_cols].drop_duplicates()

    for _, combo in unique_combos.iterrows():
        dataset = combo['dataset']
        sim_key = combo['sim_key']
        sim_rep = int(combo['sim-rep'])
        method = combo['method']
        recall_target = combo['method-recall_target'] if 'method-recall_target' in combo and not pd.isna(combo['method-recall_target']) else None
        confidence = combo['method-confidence_level'] if 'method-confidence_level' in combo and not pd.isna(combo['method-confidence_level']) else None
        sim_id = f"sim-{sim_rep}"

        # Build filters dynamically based on available parameters
        filters = [
            (df['dataset'] == dataset),
            (df['sim_key'] == sim_key),
            (df['sim-rep'] == sim_rep),
            (df['method'] == method)
        ]
        
        # Only add recall_target filter if it exists in the combo
        if recall_target is not None:
            filters.append(df['method-recall_target'] == recall_target)
            
        # Only add confidence filter if it exists in the combo
        if confidence is not None:
            filters.append(df['method-confidence_level'] == confidence)
            
        combo_data = df[pd.concat(filters, axis=1).all(axis=1)]
        
        # Sort by batch_i to ensure chronological order
        combo_data = combo_data.sort_values('batch_i')
        
        # Use method-safe_to_stop as primary indicator (according to requirements)
        stop_trigger_col = 'method-safe_to_stop'
        
        safe_stops = combo_data[combo_data[stop_trigger_col] == True]
        
        if len(safe_stops) == 0:
            # No safe stopping point found for this combination
            continue
        
        # Get the first safe stopping point
        stop_point = safe_stops.iloc[0]
        
        # Calculate achieved recall at stopping point
        n_incl_total = stop_point['n_incl']  # Total number of included studies
        n_incl_seen = stop_point['n_incl_seen']  # Number of included studies seen so far
        
        # Calculate achieved recall
        achieved_recall = n_incl_seen / n_incl_total
        
        # Get target recall
        if not pd.isna(recall_target):
            target_recall = recall_target
        else:
            # Default to 0.95 if NaN
            target_recall = 0.95
        
        # Calculate efficiency metrics for cases where target was overshot
        overshot = False
        extra_docs_ratio = 0
        
        if achieved_recall > target_recall:
            overshot = True
            
            # Find at which point we first hit the target recall
            # Sort by n_seen to ensure we find the earliest point where target was met
            combo_data_sorted = combo_data.sort_values('n_seen')
            
            first_hit_row = None

            for _, row in combo_data_sorted.iterrows():
                if n_incl_total > 0:
                    current_recall = row['n_incl_seen'] / n_incl_total
                    if current_recall >= target_recall:
                        first_hit_row = row
                        break

            if first_hit_row is not None:
                extra_docs = stop_point['n_seen'] - first_hit_row['n_seen']
                extra_docs_ratio = extra_docs / stop_point['n_total']
        
        # Modify the results structure to include recall_target and confidence
        if dataset not in results:
            results[dataset] = {}

        if method not in results[dataset]:
            results[dataset][method] = {}
            
        # Create a key for the recall target
        target_key = f"target-{target_recall}"
        if target_key not in results[dataset][method]:
            results[dataset][method][target_key] = {}
            
        # Create a key for the confidence level
        conf_key = f"conf-{confidence}" if not pd.isna(confidence) else "conf-default"
        if conf_key not in results[dataset][method][target_key]:
            results[dataset][method][target_key][conf_key] = {}

        # Store results for this specific simulation under the new nested structure
        results[dataset][method][target_key][conf_key][sim_id] = {
            'target_recall': target_recall,
            'achieved_recall': achieved_recall,
            'confidence': confidence if not pd.isna(confidence) else "default",
            'n_seen': stop_point['n_seen'],
            'n_total': stop_point['n_total'],
            'screening_proportion': stop_point['n_seen'] / stop_point['n_total'],
            'overshot': overshot,
            'extra_docs_ratio': extra_docs_ratio
        }
    
    return results

def print_results(results):
    """
    Print the evaluation results 

    results : dict
        Dictionary containing evaluation results
    """
    print("\n===== STOPPING METHOD EVALUATION =====\n")
    
    for dataset in sorted(results.keys()):
        print(f"\nDATASET: {dataset}")
        print("-" * 140)
        print(f"{'Method':<15} {'Target':<10} {'Confidence':<15} {'Simulation':<15} {'Achieved':<10} {'Screen %':<12} {'Overshot':<10} {'Extra %':<10}")
        print("-" * 140)
        
        for method in sorted(results[dataset].keys()):
            for target_key in sorted(results[dataset][method].keys()):
                for conf_key in sorted(results[dataset][method][target_key].keys()):
                    # Sort simulations by ID for consistent ordering
                    sim_ids = sorted(results[dataset][method][target_key][conf_key].keys())
                    
                    # Extract target recall and confidence from keys
                    target_recall = target_key.split('-')[1]
                    confidence = conf_key.split('-')[1]
                    
                    for sim_id in sim_ids:
                        res = results[dataset][method][target_key][conf_key][sim_id]
                        print(f"{method:<15} "
                              f"{target_recall:<10} "
                              f"{confidence:<15} "
                              f"{sim_id:<15} "
                              f"{res['achieved_recall']:.2f}      "
                              f"{res['screening_proportion']*100:.1f}%      "
                              f"{'Yes' if res['overshot'] else 'No':<10} "
                              f"{res['extra_docs_ratio']*100:.1f}%")
                    
                    # Add blank line between confidence levels
                    print()
            
            # Add blank line between methods
            print()
        
        print("\n")

def main():
    # Get the base directory using Path to ensure cross-platform compatibility
    base_dir = Path(__file__).parent.parent
    
    # Path to the results CSV file - use the location specified in the README
    csv_path = base_dir / "data" / "results" / "results.csv"
    
    if not csv_path.exists():
        logger.error(f"Results file not found at {csv_path}")
        logger.info("Please run the simulation first using the command:")
        logger.info("export PYTHONPATH=$PYTHONPATH:/path/to/stopping-methods && python simulation/main.py simulate-stopping --batch-size=100 --results_file results.csv")
        sys.exit(1)
    
    results = evaluate_stopping_methods(csv_path)
    
    print_results(results)
    
    return results

if __name__ == "__main__":
    main()