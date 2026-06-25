import os
import argparse
import copy
import json
import random
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


def _sample_combinations(combinations, num_trials, seed):
    """
    Randomly sample *num_trials* unique combinations from *combinations* without
    replacement. If num_trials >= len(combinations), all combinations are returned
    in a seeded-shuffled order (equivalent to a seeded full grid search).

    Args:
        combinations: Full list of all possible (grid) combinations.
        num_trials:   Number of trials to actually run.
        seed:         RNG seed for reproducibility.

    Returns:
        Sampled list of combinations.
    """
    rng = random.Random(seed)
    shuffled = list(combinations)
    rng.shuffle(shuffled)
    return shuffled[:num_trials]


def _apply_combo(trial_config, paths, combo):
    """Apply a combination of hyperparameter values to a (deep-copied) config dict."""
    param_desc = []
    for path, val in zip(paths, combo):
        set_nested_val(trial_config, path, val)
        param_desc.append(f"{path[-1]}={val}")
    return param_desc


def _save_best_config(base_config, paths, best_trial_row, output_dir):
    """
    Build the best config by overriding base_config with the winning trial's
    hyperparameter values and write it to <output_dir>/best_config.yaml.

    Values are cast to native Python types before serialisation so that numpy
    scalars read back from a pandas DataFrame row (e.g. np.float64, np.int64)
    don't produce Python-specific YAML tags that yaml.safe_load can't parse.
    
    Returns the path of the written file.
    """
    best_config = copy.deepcopy(base_config)
    for path in paths:
        param_key = '.'.join(path)
        raw_val = best_trial_row[param_key]
        # Coerce numpy / pandas scalar → native Python int / float / str / bool
        if hasattr(raw_val, 'item'):
            raw_val = raw_val.item()
        set_nested_val(best_config, path, raw_val)
    
    best_cfg_path = os.path.join(output_dir, "best_config.yaml")
    with open(best_cfg_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    return best_cfg_path



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
    
    method = search_cfg.get('method', 'grid')
    num_trials = search_cfg.get('num_trials', 5)
    search_seed = search_cfg.get('seed', 42)
    output_dir = search_cfg.get('output_dir', 'search_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate full Cartesian search space
    paths, values_list = get_grid_paths_and_values(grid_cfg)
    
    if not paths:
        print("No search parameters defined in search config 'grid' section.")
        return
        
    all_combinations = list(itertools.product(*values_list))

    # --- Select trial subset based on method ---
    if method == 'random':
        # Sample a reproducible subset of the grid without replacement.
        # If num_trials > total combinations fall back to using all of them.
        if num_trials >= len(all_combinations):
            print(
                f"[search] num_trials ({num_trials}) >= total combinations "
                f"({len(all_combinations)}); running all combinations in random order."
            )
        combinations = _sample_combinations(all_combinations, num_trials, seed=search_seed)
        print(
            f"Hyperparameter Search | Method: random | "
            f"Sampled Trials: {len(combinations)}/{len(all_combinations)} (seed={search_seed})"
        )
    else:
        # Full exhaustive grid search
        combinations = all_combinations
        print(
            f"Hyperparameter Search | Method: grid | "
            f"Total Trials: {len(combinations)}"
        )
    
    print("Parameters to vary:")
    for path, vals in zip(paths, values_list):
        print(f"  - {'.'.join(path)}: {vals}")
        
    results = []
    
    for idx, combo in enumerate(combinations):
        print(f"\n=================== STARTING TRIAL {idx+1}/{len(combinations)} ===================")
        
        # Clone base config and apply trial hyperparameters
        trial_config = copy.deepcopy(base_config)
        param_desc = _apply_combo(trial_config, paths, combo)
            
        trial_name = f"trial_{idx+1}_" + "_".join(param_desc)
        print(f"Trial Parameters: {', '.join(param_desc)}")
        
        # Override experiment name, log dir, and save dirs for this trial
        trial_config['logging']['experiment_name'] = trial_name
        trial_config['logging']['log_dir'] = os.path.join(output_dir, "logs")
        trial_config['logging']['tb_dir'] = os.path.join(output_dir, "runs")
        trial_config['checkpoint']['save_dir'] = os.path.join(output_dir, "checkpoints")
        
        # Disable K-Fold during search to keep individual trial times short
        trial_config['k_fold']['enabled'] = False
        
        # Run training loop for trial
        try:
            best_val = run_training(trial_config)
            
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
            
    # Save search summary CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "search_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Sort results to find best trial
    mode = base_config['checkpoint'].get('mode', 'max')
    ascending = mode == 'min'
    sorted_df = df.sort_values(by="best_val_score", ascending=ascending)
    best_trial = sorted_df.iloc[0]  # ascending=True → min at top; ascending=False → max at top
    
    print("\n" + "="*60)
    print("             HYPERPARAMETER SEARCH SUMMARY")
    print("="*60)
    print(tabulate(df, headers="keys", showindex=False, tablefmt="github"))
    print("="*60)
    print(f"BEST CONFIGURATION (Trial {best_trial['trial']}):")
    print(f"Score ({base_config['checkpoint']['monitor_metric']}): {best_trial['best_val_score']:.4f}")
    for path in paths:
        param_key = '.'.join(path)
        print(f"  - {param_key}: {best_trial[param_key]}")
    print("="*60 + "\n")
    
    # --- Save best config ---------------------------------------------------
    # Write a ready-to-use YAML that contains all base_config keys with the
    # winning hyperparameters already substituted in. Pass this directly to
    # train.py via --config to reproduce the best trial.
    best_cfg_path = _save_best_config(base_config, paths, best_trial, output_dir)
    print(f"Saved best trial config to: {best_cfg_path}")
    print(f"  → To reproduce: python train.py --config {best_cfg_path}\n")

    # Write summary markdown report
    summary_report_path = os.path.join(output_dir, "search_report.md")
    with open(summary_report_path, 'w') as f:
        f.write("# Hyperparameter Search Report\n\n")
        f.write(f"**Method**: `{method}`  \n")
        if method == 'random':
            f.write(f"**Trials sampled**: {len(combinations)}/{len(all_combinations)} (seed={search_seed})  \n")
        else:
            f.write(f"**Total trials**: {len(combinations)}  \n")
        f.write(f"**Best Trial**: Trial {best_trial['trial']} "
                f"(Score: {best_trial['best_val_score']:.4f})  \n")
        f.write(f"**Best config saved to**: `{best_cfg_path}`\n\n")
        f.write("### Trial Results Table\n\n")
        f.write(tabulate(df, headers="keys", showindex=False, tablefmt="github"))
        f.write("\n")
        
    print(f"Saved hyperparameter search report to {summary_report_path}")


if __name__ == "__main__":
    main()
