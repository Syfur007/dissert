import os
import copy
import argparse
import time
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from metrics import compute_dataset_metrics
from models import get_model
from datasets import StandardSplitDataModule
from datasets.splits import TestLoaderGuardError
from orchestration.runid import config_hash as compute_config_hash, experiment_paths
from profiling.flops import FlopsAgreementError, check_flops_agreement
from profiling.latency import measure_latency
from training.determinism import reset_recorded_nondeterminism, seed_everything
from utils.config import load_config
from utils.metrics import count_parameters
from utils.report import aggregate_repeat_reports, aggregate_seed_reports
from utils import (
    setup_logger,
    EvaluationReporter,
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
)

# Default multi-seed sweep — matches train.py's DEFAULT_SEEDS (kept as a
# separate constant rather than importing train.py, to avoid pulling in
# its training-only dependencies just for one literal).
DEFAULT_SEEDS = [7, 42, 1337]

# Matches train.py's DEFAULT_REPEATS (same duplication rationale as
# DEFAULT_SEEDS above) — must agree with whatever repeat count train.py
# actually trained, or this will look for checkpoints that don't exist.
DEFAULT_REPEATS = 3


class EnsembleModel(nn.Module):
    """
    Wraps a list of fold models behind a single nn.Module so that the rest of
    the pipeline (evaluate, check_flops_agreement, measure_latency) can treat
    an ensemble exactly like a single model. This guarantees that FLOPs, param
    counts, and throughput all reflect the *full* cost of running every fold
    model per inference, instead of just one fold's cost.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([m(x) for m in self.models], dim=0)
        return torch.mean(outputs, dim=0)


def _is_thop_profiling_buffer(key):
    """thop.profile() registers these as buffers on every submodule while
    counting FLOPs. They carry no learned information and are never read
    during a normal forward pass, so it's safe to drop them if a checkpoint
    was saved while they were still attached to the model."""
    return key.endswith("total_ops") or key.endswith("total_params")


def load_checkpoint_into(model, checkpoint_path, device, logger):
    """
    Loads a checkpoint's state dict into model, logging any key mismatches
    instead of silently swallowing them (strict=False can otherwise hide a
    checkpoint/architecture mismatch that would quietly corrupt metrics).

    Prefers EMA shadow weights over the raw weights when the checkpoint
    carries an ``ema_state`` (i.e. training used EMA). Validation during
    training runs under the EMA-averaged weights, so those — not the raw
    weights — are what actually produced the metric this checkpoint was
    saved for.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_state = checkpoint.get('ema_state')
    if ema_state and ema_state.get('shadow_state_dict'):
        logger.info(f"Checkpoint carries EMA shadow weights; using those instead of raw weights: {checkpoint_path}")
        state_dict = ema_state['shadow_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']

    # Drop thop's leftover profiling buffers (e.g. "encoder1.0.total_ops")
    # before diffing/loading -- they're not real weights and shouldn't be
    # reported as a mismatch.
    state_dict = {k: v for k, v in state_dict.items() if not _is_thop_profiling_buffer(k)}

    model_keys = set(model.state_dict().keys())
    chk_keys = set(state_dict.keys())
    missing = model_keys - chk_keys
    unexpected = chk_keys - model_keys

    if missing:
        logger.warning(f"Missing keys while loading {checkpoint_path}: {sorted(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading {checkpoint_path}: {sorted(unexpected)}")

    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate(model, dataloader, device, is_multiclass=False):
    """
    Evaluate a model (single model or EnsembleModel) on a dataset.

    Returns:
        metrics   (dict):  Averaged Dice/mIoU/HD95/ASD + per_class breakdown.
        preds_list (list): Hard per-image numpy predictions (binary or argmax).
        gts_list   (list): Raw per-image numpy ground-truth masks.
        probs_list (list): Soft probability arrays (sigmoid or softmax).
                           Shape (1, H, W) for binary, (C, H, W) for multiclass.
                           Used for ROC / PR curve computation.
    """
    preds_list = []
    gts_list   = []
    probs_list = []

    model.eval()

    with torch.no_grad():
        for images, masks, _meta in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            if not is_multiclass:
                # Binary: threshold sigmoid probabilities into a hard mask
                probs = torch.sigmoid(outputs)           # (B, 1, H, W)
                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
                probs_np = probs.cpu().numpy()           # keep (B, 1, H, W) for ROC/PR
            else:
                # Multiclass: argmax over class dimension
                probs = torch.softmax(outputs, dim=1)    # (B, C, H, W)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                probs_np = probs.cpu().numpy()           # (B, C, H, W)

            preds_list.extend([p for p in preds])
            gts_list.extend([m.cpu().numpy().astype(np.uint8) for m in masks])
            probs_list.extend([p for p in probs_np])

    # Calculate Dice, IoU, HD95, ASD, NSD, precision/recall/specificity/F2/
    # accuracy, ECE (+ per_class breakdown for multiclass) — the one
    # canonical call, per metrics/aggregate.py.
    metrics = compute_dataset_metrics(preds_list, gts_list, probs=probs_list)
    return metrics, preds_list, gts_list, probs_list


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch Segmentation Model")
    parser.add_argument("--config",      type=str, default="configs/experiment/mkunet/mkunet_t_clinicdb.yaml")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--fold",        type=int, default=None)
    parser.add_argument("--ensemble",    action="store_true")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--no-vis",      action="store_true",
                        help="Skip confusion matrix / ROC / PR curve generation.")
    parser.add_argument("--test-token",  type=str, default=None,
                        help="Pre-minted test-evaluation token (see "
                             "orchestration.ledger.LedgerWriter.issue_test_token). "
                             "Required to evaluate the test set unless --allow-test-eval is given. "
                             "Reused across every seed evaluated by this invocation.")
    parser.add_argument("--allow-test-eval", action="store_true",
                        help="Mint a fresh test-evaluation token per seed evaluated (each records "
                             "its own Test_Evals ledger row) instead of requiring a pre-minted "
                             "--test-token.")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override logging.experiment_name (and therefore both the log dir "
                             "and, unless --checkpoint is given explicitly, the checkpoint "
                             "lookup dir) from the config. Lets a caller (e.g. "
                             "scripts/reproduce.sh) point a run at a scoped name instead of "
                             "silently reusing — and overwriting the logs/report.json of — "
                             "whatever real experiment already used the config's own name.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None,
                        help="Evaluate exactly this one seed, bypassing the default 3-seed sweep "
                             "and combined report — e.g. to check one particular seed manually.")
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"Evaluate exactly these seeds, then write the combined averaged "
                             f"report (default: {DEFAULT_SEEDS}).")
    parser.add_argument("--repeats", type=int, default=None,
                        help=f"Number of identical re-runs per seed to evaluate (must match what "
                             f"train.py trained; default: {DEFAULT_REPEATS}). Ignored when --seed "
                             f"bypasses the sweep.")
    return parser.parse_args()


def evaluate_one(config: dict, args, seed: int, repeat: Optional[int] = None):
    """Runs the full evaluation pipeline for one (seed, repeat): resolves
    that run's checkpoint(s), evaluates on the test set, writes
    report.json/.md under its eval/ dir.

    Args:
        repeat: Nth identical re-run of this seed (see
            ``train.run_training``'s ``repeat`` arg) — ``None`` (default)
            evaluates the single, non-repeated checkpoint at the legacy
            unsuffixed path.

    Returns the report JSON path, or ``None`` if evaluation could not
    proceed (missing test token/loader/checkpoint) — logged, not raised,
    matching this module's existing "log it and return" style so one bad
    run doesn't take down the rest of a multi-seed/multi-repeat call.
    """
    config = copy.deepcopy(config)
    config['training']['seed'] = seed

    training_cfg = config['training']
    dataset_cfg  = config['dataset']
    kfold_cfg    = config.get('k_fold', {})
    chk_cfg      = config.get('checkpoint', {})
    log_cfg      = config.get('logging', {})

    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")
    reset_recorded_nondeterminism()
    seed_everything(training_cfg["seed"])

    # ── Output layout ──────────────────────────────────────────────────
    base_exp   = log_cfg['experiment_name']
    output_dir = config.get("output_dir", "outputs/experiments")
    resolved_config_hash = compute_config_hash(config)
    # fold=None here: logs/ and eval/ are shared across all folds of this
    # experiment (same as train.py) — only the checkpoint *read* path below
    # is fold-scoped. repeat threads through so a repeated run's eval/
    # report.json lands in its own independent {hash7}-s{seed}-r{repeat}/
    # tree, same as its checkpoints do.
    exp_paths = experiment_paths(output_dir, base_exp, resolved_config_hash, seed, repeat=repeat)

    # ── Logging ────────────────────────────────────────────────────────
    logger, exp_log_dir = setup_logger(exp_paths["logs"], "eval")
    logger.info(f"Using device: {device}")
    logger.info(f"Eval log dir: {exp_log_dir}")

    # ── Test-loader guard token ────────────────────────────────────────
    # Touching the test set requires a token minted by
    # orchestration.ledger.LedgerWriter.issue_test_token() — either a
    # pre-minted one (--test-token, e.g. from an orchestrated sweep, reused
    # across every seed this invocation evaluates) or a freshly self-issued
    # one per seed (--allow-test-eval, for a manual run); either way a
    # Test_Evals ledger row is left behind.
    if args.test_token:
        test_token = args.test_token
    elif args.allow_test_eval:
        from orchestration.ledger import LedgerWriter
        repeat_tag = f"-r{repeat}" if repeat is not None else ""
        test_token = LedgerWriter().issue_test_token(
            run_id=f"manual-eval-{base_exp}-s{seed}{repeat_tag}",
            config_hash=resolved_config_hash,
        )
    else:
        logger.error(
            "Evaluating the test set requires --test-token <TOKEN> (from an "
            "orchestrated sweep) or --allow-test-eval (mints one for this "
            "manual run). Refusing to proceed without a recorded test-eval token."
        )
        return None

    # Init Datamodule
    dm = StandardSplitDataModule(config)
    try:
        test_loader = dm.get_test_loader(test_token)
    except TestLoaderGuardError as exc:
        logger.error(str(exc))
        return None

    if test_loader is None:
        logger.error("No test set filenames or separate test directory was found in configuration.")
        return None

    logger.info(f"Test samples found: {len(test_loader.dataset)}")

    # Init Model structure
    model_cfg     = config['model']
    is_multiclass = model_cfg['out_channels'] > 1
    class_names   = dataset_cfg.get('class_names', None)

    # Determine which checkpoints to load. Checkpoints are fold-scoped
    # (checkpoints/fold{N}/best.pth) — resolve per-fold below.
    def _checkpoints_dir(fold=None):
        return experiment_paths(output_dir, base_exp, resolved_config_hash, seed, fold, repeat)["checkpoints"]

    if args.ensemble:
        # Load all fold checkpoints for ensembling
        n_splits = kfold_cfg.get('n_splits', 5)
        logger.info(f"Loading ensemble models from all {n_splits} folds...")

        fold_models = []
        loaded_checkpoint_paths = []
        for f in range(n_splits):
            fold_chk_path = os.path.join(_checkpoints_dir(f), "best.pth")
            if os.path.exists(fold_chk_path):
                model_f = get_model(**model_cfg).to(device)
                model_f = load_checkpoint_into(model_f, fold_chk_path, device, logger)
                fold_models.append(model_f)
                loaded_checkpoint_paths.append(fold_chk_path)
                logger.info(f"Loaded fold {f} from {fold_chk_path}")
            else:
                logger.warning(f"Could not find checkpoint for fold {f} at {fold_chk_path}. Skipping.")

        if not fold_models:
            logger.error("No fold checkpoints could be loaded for ensembling.")
            return None

        model = EnsembleModel(fold_models).to(device)
    else:
        model = get_model(**model_cfg).to(device)

        if args.checkpoint:
            chk_path = args.checkpoint
        elif args.fold is not None:
            chk_path = os.path.join(_checkpoints_dir(args.fold), "best.pth")
        else:
            chk_path = os.path.join(_checkpoints_dir(), "best.pth")
            if not os.path.exists(chk_path):
                chk_path = os.path.join(_checkpoints_dir(0), "best.pth")

        if not os.path.exists(chk_path):
            logger.error(f"Checkpoint file not found: {chk_path}")
            return None

        logger.info(f"Loading weights from checkpoint: {chk_path}")
        model = load_checkpoint_into(model, chk_path, device, logger)
        loaded_checkpoint_paths = chk_path

    # Profile complexity: analytic FLOPs (fvcore-agreement-checked, Phase 10 —
    # profiling/flops.py) + trainable param count. A disagreement is a real
    # correctness signal (see that module's docstring), not blocked on here —
    # eval.py still reports whatever FLOPs figure the check did compute, with
    # a clear warning, rather than aborting evaluation over a profiling gap.
    input_shape = (model_cfg['in_channels'], dataset_cfg['img_height'], dataset_cfg['img_width'])
    params = count_parameters(model)
    try:
        flops_result = check_flops_agreement(model, input_shape, tolerance=0.05)
        flops = flops_result["reported_total"]
    except FlopsAgreementError as exc:
        logger.warning(f"FLOPs agreement check failed: {exc}")
        flops = 0

    # Measure evaluation throughput (batch=1, spec §14 protocol: >=50
    # warm-up, >=200 timed runs — profiling/latency.py)
    logger.info("Measuring inference throughput...")
    throughput = measure_latency(model, input_shape, device, batch_size=1, num_warmup=50, num_runs=200)["throughput_ips"]

    # Build reporter early (latency measured before the eval loop)
    reporter = EvaluationReporter(config, args, logger)
    reporter.set_model_info(
        model            = model,
        flops            = flops,
        params           = params,
        throughput       = throughput,
        checkpoint_path  = loaded_checkpoint_paths if args.ensemble else chk_path,
        measure_latency  = True,
    )

    # ── Evaluation loop ────────────────────────────────────────────────
    logger.info("Starting test set evaluation...")
    start_eval_time = time.time()

    metrics, preds_list, gts_list, probs_list = evaluate(
        model, test_loader, device, is_multiclass=is_multiclass
    )

    eval_duration = time.time() - start_eval_time
    logger.info(f"Evaluation finished in {eval_duration:.2f} seconds.")

    # Log macro metrics
    logger.info(
        f"Dice: {metrics['dice']:.4f} | mIoU: {metrics['miou']:.4f} | "
        f"HD95: {metrics['hd95']:.2f} | ASD: {metrics['asd']:.2f}"
    )

    # Log per-class breakdown if available
    pc = metrics.get("per_class", {})
    if pc:
        class_dice = pc.get("dice", [])
        class_iou  = pc.get("iou",  [])
        lines = []
        for c in range(len(class_dice)):
            name = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
            lines.append(f"  {name}: Dice={class_dice[c]:.4f}  IoU={class_iou[c]:.4f}")
        logger.info("Per-class metrics:\n" + "\n".join(lines))

    # ── Visualisations (confusion matrix, ROC, PR) ─────────────────────
    if not args.no_vis:
        vis_dir = os.path.join(exp_paths["eval"], "curves")
        os.makedirs(vis_dir, exist_ok=True)

        # Confusion matrix (from hard predictions)
        try:
            cm_path = os.path.join(vis_dir, "confusion_matrix.png")
            save_confusion_matrix(
                preds_list, gts_list, cm_path,
                class_names=class_names,
                normalize=True,
                title=f"Confusion Matrix — {log_cfg['experiment_name']}",
            )
            logger.info(f"Saved confusion matrix → {cm_path}")
        except Exception as exc:
            logger.warning(f"Could not save confusion matrix: {exc}")

        # ROC curves (from soft probabilities)
        try:
            roc_path = os.path.join(vis_dir, "roc_curve.png")
            save_roc_curve(
                probs_list, gts_list, roc_path,
                class_names=class_names,
                title=f"ROC Curve — {log_cfg['experiment_name']}",
            )
            logger.info(f"Saved ROC curve → {roc_path}")
        except Exception as exc:
            logger.warning(f"Could not save ROC curve: {exc}")

        # PR curves (from soft probabilities)
        try:
            pr_path = os.path.join(vis_dir, "pr_curve.png")
            save_pr_curve(
                probs_list, gts_list, pr_path,
                class_names=class_names,
                title=f"Precision-Recall Curve — {log_cfg['experiment_name']}",
            )
            logger.info(f"Saved PR curve → {pr_path}")
        except Exception as exc:
            logger.warning(f"Could not save PR curve: {exc}")

    # ── Report ─────────────────────────────────────────────────────────
    reporter.set_eval_results(
        base_metrics    = metrics,
        num_samples     = len(test_loader.dataset),
        eval_duration_s = eval_duration,
        is_multiclass   = is_multiclass,
    )

    reporter.print_console()
    reporter.save(
        report_dir      = exp_paths["eval"],
        # filename_prefix = log_cfg['experiment_name'],
    )

    ensemble_tag = "ensemble_" if args.ensemble else ""
    return os.path.join(exp_paths["eval"], f"{ensemble_tag}report.json")


def main():
    args = _parse_args()
    config = load_config(args.config)

    if args.dataset_dir is not None:
        config['dataset']['root'] = args.dataset_dir
    if args.experiment_name is not None:
        config['logging']['experiment_name'] = args.experiment_name

    if args.seed is not None:
        # Single explicit seed — bypasses the default sweep and combined
        # report entirely.
        evaluate_one(config, args, args.seed)
        return

    # Default: evaluate every seed in DEFAULT_SEEDS (or --seeds) x every
    # repeat in DEFAULT_REPEATS (or --repeats), in process. Two-level
    # aggregation: repeats are averaged within a seed first (noise
    # surviving fixed seeding — see aggregate_repeat_reports), then those
    # already-denoised per-seed reports are averaged across seeds exactly
    # as before repeats existed.
    seeds = args.seeds or DEFAULT_SEEDS
    n_repeats = args.repeats if args.repeats is not None else DEFAULT_REPEATS
    # n_repeats<=1 keeps the legacy unsuffixed single-run path (None,) —
    # same convention train.py's default sweep uses.
    repeats = (None,) if n_repeats <= 1 else list(range(n_repeats))

    seed_reports = []  # [(seed, path to that seed's own report)]
    for s in seeds:
        per_repeat = []
        for r in repeats:
            report_path = evaluate_one(config, args, s, repeat=r)
            if report_path is not None:
                per_repeat.append((r, report_path))

        if not per_repeat:
            print(f"Seed {s}: no repeat evaluation succeeded — skipping this seed.")
            continue
        if len(per_repeat) < len(repeats):
            print(f"Seed {s}: only {len(per_repeat)}/{len(repeats)} repeat evals succeeded.")

        if repeats == (None,):
            # No repeat axis in use — that single report already *is*
            # this seed's report, same as before repeats existed.
            seed_reports.append((s, per_repeat[0][1]))
        else:
            seed_report_path = aggregate_repeat_reports(
                config, [p for _, p in per_repeat], [r for r, _ in per_repeat], seed=s,
            )
            print(f"Seed {s}: combined {len(per_repeat)} repeat(s) → {seed_report_path}")
            seed_reports.append((s, seed_report_path))

    if not seed_reports:
        print("No seed evaluation succeeded — skipping combined report.")
        return
    if len(seed_reports) < len(seeds):
        print(f"Only {len(seed_reports)}/{len(seeds)} seed evals succeeded; "
              f"combined report covers just those.")

    combined_path = aggregate_seed_reports(
        config, [p for _, p in seed_reports], [s for s, _ in seed_reports]
    )
    print(f"Combined report ({len(seed_reports)} seed(s)) → {combined_path}")


if __name__ == "__main__":
    main()