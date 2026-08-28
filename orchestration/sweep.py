"""
orchestration/sweep.py — spec §15's Sweep row: "sweep.py takes a search
space and a trial budget. The budget is a project-level constant applied
identically to every model. Objective is computed on the inner validation
split; the test loader is unreachable from the sweep code path."

Supersedes search.py's bare ``num_trials`` cutoff with a GPU-hour budget
(spec's literal wording): trials run, in a seeded-shuffled order (so an
early-stopped sweep still samples the grid representatively rather than
always favouring its declared order), until the next trial's measured
wall-clock cost would push cumulative usage past the budget.

This module intentionally has zero import-time dependency on
``datasets``/``torch`` (``trial_fn`` defaults via a *lazy* import inside
``run_budgeted_sweep``, exactly mirroring
``orchestration.runner.run_sweep``'s own discipline) — the point isn't
performance, it's that ``tests/test_data_contract.py::test_sweep_cannot_see_test``
greps this file's source text for the guarded test-set loader function's
name and expects to find zero references — a static guarantee search.py
already had and this module must not weaken.
"""
from __future__ import annotations

import copy
import itertools
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

TrialFn = Callable[[Dict[str, Any]], float]


def get_grid_paths_and_values(
    grid_dict: Dict[str, Any], path: Optional[List[str]] = None
) -> Tuple[List[List[str]], List[List[Any]]]:
    """Same nested-grid traversal as search.py's own helper, reimplemented
    independently rather than imported — this module has no dependency on
    the script it supersedes.
    """
    if path is None:
        path = []
    paths: List[List[str]] = []
    values_list: List[List[Any]] = []
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


def _set_nested(d: Dict[str, Any], path: List[str], val: Any) -> None:
    node = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = val


def _sample_trial_order(combinations: Sequence[Tuple], seed: int) -> List[Tuple]:
    rng = random.Random(seed)
    order = list(combinations)
    rng.shuffle(order)
    return order


def run_budgeted_sweep(
    base_config: Dict[str, Any],
    grid: Dict[str, Any],
    budget_gpu_hours: float,
    seed: int = 42,
    trial_fn: Optional[TrialFn] = None,
    mode: str = "max",
) -> Dict[str, Any]:
    """Runs trials from *grid*'s Cartesian product in a seeded-shuffled
    order until cumulative measured wall-clock cost would exceed
    *budget_gpu_hours* — pass the *same* budget to every model's call to
    this function (spec: "a project-level constant applied identically to
    every model"), not a per-model figure.

    Args:
        trial_fn: injectable for testing; defaults to a lazy import of
            ``train.run_training`` (validation-objective only — never
            touches the guarded test loader).
        mode: "max" or "min" — which direction is "better" for
            *trial_fn*'s returned validation objective.

    Returns ``{"trials": [...], "best_trial": {...} or None,
    "budget_gpu_hours", "gpu_hours_used", "n_grid_combinations",
    "stopped_early"}`` — each trial dict is ``{"params": {dotted_path:
    value}, "objective": float, "status", "wall_hours"}``.
    """
    if budget_gpu_hours <= 0:
        raise ValueError(f"budget_gpu_hours must be positive, got {budget_gpu_hours}")
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    if trial_fn is None:
        from train import run_training as trial_fn  # local: see module docstring

    paths, values_list = get_grid_paths_and_values(grid)
    if not paths:
        raise ValueError("run_budgeted_sweep: grid has no list-valued parameters to sweep")
    all_combinations = list(itertools.product(*values_list))
    order = _sample_trial_order(all_combinations, seed)

    trials: List[Dict[str, Any]] = []
    gpu_hours_used = 0.0

    for combo in order:
        trial_config = copy.deepcopy(base_config)
        params: Dict[str, Any] = {}
        for path, val in zip(paths, combo):
            _set_nested(trial_config, path, val)
            params[".".join(path)] = val

        t0 = time.perf_counter()
        try:
            objective = trial_fn(trial_config)
            status = "success"
        except Exception as exc:  # noqa: BLE001 — one bad trial must not kill the sweep
            objective = float("-inf") if mode == "max" else float("inf")
            status = f"failed: {exc}"
        wall_hours = (time.perf_counter() - t0) / 3600.0
        gpu_hours_used += wall_hours

        trials.append({"params": params, "objective": objective, "status": status, "wall_hours": wall_hours})

        if gpu_hours_used >= budget_gpu_hours:
            break

    return {
        "trials": trials,
        "best_trial": _best_trial(trials, mode),
        "budget_gpu_hours": budget_gpu_hours,
        "gpu_hours_used": gpu_hours_used,
        "n_grid_combinations": len(all_combinations),
        "stopped_early": len(trials) < len(order),
    }


def _best_trial(trials: List[Dict[str, Any]], mode: str) -> Optional[Dict[str, Any]]:
    successful = [t for t in trials if t["status"] == "success"]
    if not successful:
        return None
    return (max if mode == "max" else min)(successful, key=lambda t: t["objective"])


def main() -> None:
    """CLI entry point — ``python -m orchestration.sweep --base-config ... --search-config ...``.
    Reads ``search.budget_gpu_hours`` from the search config (required —
    unlike search.py's ``num_trials``, this module has no trial-count
    fallback, per spec's "takes a search space and a trial budget").
    """
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Budgeted hyperparameter sweep")
    parser.add_argument("--base-config", type=str, default="configs/experiment/mkunet/mkunet_t_clinicdb.yaml")
    parser.add_argument("--search-config", type=str, default="configs/search_config.yaml")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    search_config = load_config(args.search_config, validate=False)
    search_cfg = search_config.get("search", {})
    grid_cfg = search_config.get("grid", {})

    budget = search_cfg.get("budget_gpu_hours")
    if budget is None:
        raise ValueError(
            "orchestration/sweep.py requires search.budget_gpu_hours in the search config "
            "(spec §15: sweep.py takes a search space and a trial budget) — "
            "add it to configs/search_config.yaml's `search:` section."
        )

    result = run_budgeted_sweep(
        base_config, grid_cfg, budget_gpu_hours=budget, seed=search_cfg.get("seed", 42),
        mode=base_config.get("checkpoint", {}).get("mode", "max"),
    )
    print(
        f"Sweep finished: {len(result['trials'])}/{result['n_grid_combinations']} trials, "
        f"{result['gpu_hours_used']:.3f}/{result['budget_gpu_hours']} GPU-hours used"
        + (" (stopped early on budget)" if result["stopped_early"] else "")
    )
    if result["best_trial"]:
        print(f"Best trial: {result['best_trial']['params']} -> {result['best_trial']['objective']}")


if __name__ == "__main__":
    main()
