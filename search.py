import os
import argparse
import copy
import json
import itertools
import yaml
import pandas as pd
from tabulate import tabulate

from train import run_training

def get_grid_paths_and_values(grid_dict, path=None):
    """
    Traverse a nested grid dictionary and return paths to parameters and their lists of values.
    
    Returns:
        paths (list of list): Key paths to parameters, e.g. [["training", "lr"], ["dataset", "batch_size"]]
        values (list of list): The lists of values to grid search over.
    """
    if path is None:
        path = []
    paths = []
    values_list = []
    
    for k, v in grid_dict.items():
        current_path = path + [k]
        if isinstance(v, dict):
            sub_paths, sub_vals = get_grid_paths_and_values(v, current_path)
            paths.extend(sub_paths)
            values_list.extend(sub_vals)
        elif isinstance(v, list):
            paths.append(current_path)
            values_list.append(v)
            
    return paths, values_list

def set_nested_val(dict_obj, path, val):
    """Set value in a nested dictionary using a key path list."""
    d = dict_obj
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = val

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search Runner")
    parser.add_argument("--base-config", type=str, default="configs/base_config.yaml", help="Path to base configuration file")
    parser.add_argument("--search-config", type=str, default="configs/search_config.yaml", help="Path to search configuration file")
    args = parser.parse_args()
    
    # Load configs
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
        
    with open(args.search_config, 'r') as f:
        search_config = yaml.safe_load(f)
        
    search_cfg = search_config.get('search', {})
    grid_cfg = search_config.get('grid', {})
    
    output_dir = search_cfg.get('output_dir', 'search_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate search space combinations
    paths, values_list = get_grid_paths_and_values(grid_cfg)
    
    if not paths:
        print("No search parameters defined in search config 'grid' section.")
        return
        
    combinations = list(itertools.product(*values_list))
    
    print(f"Hyperparameter Search | Method: {search_cfg.get('method', 'grid')} | Total Trials: {len(combinations)}")
    print("Parameters to vary:")
    for path, vals in zip(paths, values_list):
        print(f"  - {'.'.join(path)}: {vals}")
        
    results = []
    
    for idx, combo in enumerate(combinations):
        print(f"\n=================== STARTING TRIAL {idx+1}/{len(combinations)} ===================")
        
        # Clone base config
        trial_config = copy.deepcopy(base_config)
        
        # Apply parameters
        param_desc = []
        for path, val in zip(paths, combo):
            set_nested_val(trial_config, path, val)
            param_desc.append(f"{path[-1]}={val}")
            
        trial_name = f"trial_{idx+1}_" + "_".join(param_desc)
        print(f"Trial Parameters: {', '.join(param_desc)}")
        
        # Override experiment name, log dir, and save dirs for this trial
        trial_config['logging']['experiment_name'] = trial_name
        trial_config['logging']['log_dir'] = os.path.join(output_dir, "logs")
        trial_config['logging']['tb_dir'] = os.path.join(output_dir, "runs")
        trial_config['checkpoint']['save_dir'] = os.path.join(output_dir, "checkpoints")
        
        # Run training loop for trial
        try:
            # We run a single training process (or single fold, or k-fold according to base config)
            # To make hyperparameter search fast, we can run single train runs
            # We set K-Fold CV to false for search trials by default to avoid huge training times
            trial_config['k_fold']['enabled'] = False
            best_val = run_training(trial_config)
            
            # Save results
            trial_result = {
                "trial": idx + 1,
                "best_val_score": best_val,
                "status": "success"
            }
            for path, val in zip(paths, combo):
                trial_result['.'.join(path)] = val
                
            results.append(trial_result)
        except Exception as e:
            print(f"Trial {idx+1} failed with error: {e}")
            trial_result = {
                "trial": idx + 1,
                "best_val_score": float('-inf') if base_config['checkpoint']['mode'] == 'max' else float('inf'),
                "status": f"failed: {e}"
            }
            for path, val in zip(paths, combo):
                trial_result['.'.join(path)] = val
            results.append(trial_result)
            
    # Save search summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "search_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Sort results to find best trial
    mode = base_config['checkpoint'].get('mode', 'max')
    sorted_df = df.sort_values(by="best_val_score", ascending=(mode == 'min'))
    best_trial = sorted_df.iloc[0] if mode == 'min' else sorted_df.iloc[-1]
    
    print("\n" + "="*50)
    print("             HYPERPARAMETER SEARCH SUMMARY")
    print("="*50)
    print(tabulate(df, headers="keys", showindex=False, tablefmt="github"))
    print("="*50)
    print(f"BEST CONFIGURATION (Trial {best_trial['trial']}):")
    print(f"Score ({base_config['checkpoint']['monitor_metric']}): {best_trial['best_val_score']:.4f}")
    for path in paths:
        param_key = '.'.join(path)
        print(f"  - {param_key}: {best_trial[param_key]}")
    print("="*50 + "\n")
    
    # Write summary markdown report
    summary_report_path = os.path.join(output_dir, "search_report.md")
    with open(summary_report_path, 'w') as f:
        f.write("# Hyperparameter Search Report\n\n")
        f.write(f"**Best Trial**: Trial {best_trial['trial']} (Score: {best_trial['best_val_score']:.4f})\n\n")
        f.write("### Trial Results Table\n\n")
        f.write(tabulate(df, headers="keys", showindex=False, tablefmt="github"))
        f.write("\n")
        
    print(f"Saved hyperparameter search report to {summary_report_path}")

if __name__ == "__main__":
    main()
