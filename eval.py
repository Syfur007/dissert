import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import get_model
from datasets import StandardSplitDataModule
from utils import (
    setup_logger,
    compute_dataset_metrics,
    log_model_summary,
    measure_throughput,
    EvaluationReporter,
)


class EnsembleModel(nn.Module):
    """
    Wraps a list of fold models behind a single nn.Module so that the rest of
    the pipeline (evaluate, get_flops_and_params, measure_throughput) can treat
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
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
        metrics (dict): Averaged Dice/mIoU/HD95/ASD across the test set.
        preds_list (list): Raw per-image numpy predictions (for extended metrics).
        gts_list (list): Raw per-image numpy ground-truth masks.
    """
    preds_list = []
    gts_list = []

    model.eval()

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            # Prepare predictions and ground truths for metrics
            if not is_multiclass:
                # Binary: threshold sigmoid probabilities into a hard mask
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
            else:
                # Multiclass: take the most likely class per pixel so shapes
                # match the (H, W) integer ground-truth masks. Without this
                # argmax, preds would stay as (C, H, W) softmax probabilities,
                # which is shape/type-incompatible with gts_list and breaks
                # Dice/IoU/HD95/ASD computation.
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)

            preds_list.extend([p for p in preds])
            gts_list.extend([m.cpu().numpy().astype(np.uint8) for m in masks])

    # Calculate Dice, IoU, HD95, ASD
    metrics = compute_dataset_metrics(preds_list, gts_list)
    return metrics, preds_list, gts_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to model checkpoint")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold checkpoint to evaluate")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble evaluation of all K-folds")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.dataset_dir is not None:
        config['dataset']['root'] = args.dataset_dir

    training_cfg = config['training']
    dataset_cfg = config['dataset']
    kfold_cfg = config.get('k_fold', {})
    chk_cfg = config.get('checkpoint', {})
    log_cfg = config.get('logging', {})

    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_cfg['log_dir'], f"{log_cfg['experiment_name']}_eval")
    logger.info(f"Using device: {device}")

    # Init Datamodule
    dm = StandardSplitDataModule(config)
    test_loader = dm.get_test_loader()

    if test_loader is None:
        logger.error("No test set filenames or separate test directory was found in configuration.")
        return

    logger.info(f"Test samples found: {len(test_loader.dataset)}")

    # Init Model structure
    model_cfg = config['model']
    is_multiclass = model_cfg['out_channels'] > 1

    # Determine which checkpoints to load
    checkpoint_dir = os.path.join(chk_cfg.get('save_dir', 'checkpoints'), log_cfg['experiment_name'])

    if args.ensemble:
        # Load all fold checkpoints for ensembling
        n_splits = kfold_cfg.get('n_splits', 5)
        logger.info(f"Loading ensemble models from all {n_splits} folds...")

        fold_models = []
        loaded_checkpoint_paths = []
        for f in range(n_splits):
            fold_chk_path = os.path.join(checkpoint_dir, f"best_fold{f}.pth")
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
            return

        # Wrap all fold models in a single module so FLOPs/params/throughput
        # and evaluate() all measure the actual ensemble cost, not just one fold.
        model = EnsembleModel(fold_models).to(device)
    else:
        # Load a single model
        model = get_model(**model_cfg).to(device)

        # Determine path
        if args.checkpoint:
            chk_path = args.checkpoint
        elif args.fold is not None:
            chk_path = os.path.join(checkpoint_dir, f"best_fold{args.fold}.pth")
        else:
            # Fallback to standard non-k-fold best or fold0 best
            chk_path = os.path.join(checkpoint_dir, "best.pth")
            if not os.path.exists(chk_path):
                chk_path = os.path.join(checkpoint_dir, "best_fold0.pth")

        if not os.path.exists(chk_path):
            logger.error(f"Checkpoint file not found: {chk_path}")
            return

        logger.info(f"Loading weights from checkpoint: {chk_path}")
        model = load_checkpoint_into(model, chk_path, device, logger)

    # Profile complexity — returns (flops, params) and logs + writes model_summary.txt
    input_shape = (1, model_cfg['in_channels'], dataset_cfg['img_height'], dataset_cfg['img_width'])
    flops, params = log_model_summary(model, input_shape, logger, log_dir=log_cfg.get('log_dir'))

    # Measure evaluation throughput (images/sec); also reflects full ensemble cost when --ensemble is set
    logger.info("Measuring inference throughput...")
    throughput = measure_throughput(model, test_loader, device)

    # Build the reporter early so latency measurement happens before the eval loop.
    # checkpoint_path is a list of fold files in ensemble mode -- passing the
    # directory here previously made get_model_disk_size() report the
    # directory inode size instead of the combined checkpoint size.
    reporter = EvaluationReporter(config, args, logger)
    reporter.set_model_info(
        model=model,
        flops=flops,
        params=params,
        throughput=throughput,
        checkpoint_path=loaded_checkpoint_paths if args.ensemble else chk_path,
        measure_latency=True,
    )

    # Run evaluation
    logger.info("Starting test set evaluation...")
    start_eval_time = time.time()

    metrics, preds_list, gts_list = evaluate(model, test_loader, device, is_multiclass=is_multiclass)

    eval_duration = time.time() - start_eval_time
    logger.info(f"Evaluation finished in {eval_duration:.2f} seconds.")

    # Populate remaining results and render reports
    reporter.set_eval_results(
        base_metrics=metrics,
        preds=preds_list,
        gts=gts_list,
        num_samples=len(test_loader.dataset),
        eval_duration_s=eval_duration,
        is_multiclass=is_multiclass,
    )

    reporter.print_console()
    reporter.save(
        report_dir=log_cfg['log_dir'],
        filename_prefix=log_cfg['experiment_name'],
    )


if __name__ == "__main__":
    main()