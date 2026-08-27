# Changelog

Tracks framework-level changes made while bringing `dissert` into compliance with
`Technical_Framework_Spec.md`, per `IMPLEMENTATION_PLAN.md`. Not a user-facing release log.

## Phase 0 — torch 1.11 → 1.13 environment bump (2026-08-27)

- `torch==1.11.0`/`torchvision==0.12.0` → `torch==1.13.1`/`torchvision==0.14.1`. All other
  pinned packages (`numpy`, `albumentations`, `timm`, `transformers`, `huggingface-hub`, ...)
  resolved cleanly against the new torch pin with no forced bumps.
- Added `pydantic>=2,<3`, `statsmodels`, `fvcore`, `captum` (installed and verified importable),
  and `mamba-ssm==1.0.1` / `causal-conv1d==1.1.1` (pinned for the Phase 6 Mamba/VSS model family;
  **not installable on the dev sandbox used for this work**, which has no `nvcc`/CUDA toolkit —
  `pip install` fails at metadata generation with `NameError: name 'bare_metal_version' is not
  defined`, the expected failure mode of their setup.py's CUDA-toolkit probe when `nvcc` is
  absent. Confirmed prebuilt wheels exist for CUDA 11.8 + torch 1.13 + Python 3.8 on both
  projects' GitHub Releases — see `requirements.txt`'s comment block for exact URLs — for
  installation on the actual GPU training machine. Until then, Phase 6 must use the
  `models/auxiliary/ss2d_ref.py` pure-PyTorch fallback in this sandbox.)
- Removed `torchaudio==0.11.0+cu113` from the `thesis` conda env: unpinned in `requirements.txt`,
  unimported anywhere in the codebase, and the only package `pip check` flagged as conflicting
  with the new torch pin. Left `mmcv-full`/`addict`/`yapf` installed but undeclared/unused
  (present in the env, imported nowhere, nothing depends on them) — not removed since they were
  pre-existing and not blocking anything.
- Fixed `compose: ["../../base_config.yaml"]` in 11 experiment configs under
  `configs/experiment/{gmkunet,mkunet}/*.yaml` — one directory level too many, resolved to the
  nonexistent `configs/base_config.yaml` instead of `configs/experiment/base_config.yaml`. This
  was a pre-existing bug (unrelated to the torch bump) that made every one of those configs
  unloadable; discovered while trying to run this phase's own reproducibility smoke test.
- **Reproducibility smoke test** (`checkpoints/gmkunet_t_clinicdb/best_fold0.pth`,
  `configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml`, fold 0, ClinicDB test split, CPU — this
  sandbox has no GPU): `eval.py` under torch 1.11.0+cu113 vs. torch 1.13.1+cpu produced
  bit-identical results — Dice 0.7816, mIoU 0.6881, HD95 35.39px, ASD 13.31px, Precision 0.7636,
  Recall 0.8640 — in both runs. No drift beyond expected latency/throughput timing noise.
- **Training smoke test**: `mkunet_t_clinicdb`, fold 0, 2 epochs, CPU, torch 1.13.1 — dataloader
  caching, AMP CUDA-guard (correctly no-ops on CPU), optimizer/scheduler, checkpointing,
  EarlyStopping, and callbacks all ran without error; Dice rose 0.163 → 0.258 across the 2 epochs
  from a fresh init, as expected for a from-scratch run (not a resume, so not a like-for-like
  comparison to a historical checkpoint — the eval-side smoke test above is the actual
  before/after reproducibility check).
- **EMCAD / PVTv2**: model builds and runs a forward pass cleanly under torch 1.13. Could not
  verify the actual `torch.load(...)` pretrained-checkpoint path end-to-end — no
  `pretrained_pth/pvt/*.pth` file exists in this sandbox (gitignored, never downloaded here); the
  loader correctly warns and falls back to random init when the file is absent. No code change
  needed regardless — torch 1.13 is well below the 2.6 `weights_only` default flip.
- `requirements.txt`, `env/environment.lock` (new), `README.md` updated accordingly.

**Not verified in this sandbox** (no GPU, no CUDA toolkit, no internet-independent build tools):
CUDA-build install of torch/torchvision, `mamba-ssm`/`causal-conv1d` import or kernel execution,
real (non-random-init) PVTv2 checkpoint loading, any GPU-only behavior. These need to be repeated
on the actual training machine before treating Phase 0 as fully closed there.

## Phase 1 — Config schema, run identity, manifest, determinism, orchestration core (2026-08-27)

- **New `orchestration/` package**: `schema.py` (pydantic `Config`/`SearchSweepConfig` models
  mirroring every section of `configs/base.yaml`; `validate_config()`), `runid.py`
  (`config_hash()` — SHA1 over the resolved config with `training.seed` excluded;
  `run_id()` → `R-{hash[:7]}-s{seed}-f{fold}`, with `f-` for a non-CV run's fold segment),
  `manifest.py` (`RunManifest`/`build_manifest()` — resolved config, config hash, git commit +
  dirty flag, env hash, hardware, timing, `nondeterministic_ops`, written to
  `artifacts/runs/<run_id>/manifest.json`), `ledger.py` (`LedgerWriter`, CSV-backed
  Runs/Compute/Test_Evals/Stats tables — only Runs/Compute are populated yet), `runner.py`
  (`run_sweep()` — expands seed×fold, idempotent `status: done` skip, manifest+ledger wrapping
  around `train.run_training()`, one bad run doesn't stop the sweep).
- **`utils/config.py`**: `load_config()` now pipes its merged dict through
  `orchestration.schema.validate_config()` by default (`validate=False` opts out for a fragment
  loaded standalone) — an unknown key, a missing required key, or a wrong-typed value anywhere in
  the config tree now raises `pydantic.ValidationError` at load time instead of a confusing
  `KeyError`/silent no-op deep in train.py/eval.py.
- **New `training/determinism.py`**: single `seed_everything(seed)` (python/numpy/torch/cuda +
  `torch.use_deterministic_algorithms(True, warn_only=True)`), replacing train.py's local
  `set_seed()`. Captures any non-deterministic-op warning torch emits into
  `get_recorded_nondeterminism()` rather than silently dropping it. `eval.py` — which previously
  had **zero** seeding — now calls it too.
- **`utils/checkpoint.py`**: `CheckpointManager` now accepts `config_hash`/`run_id` and embeds
  them plus the current git commit into every saved `last*.pth`/`best*.pth`. `train.py`'s
  `run_training()` gained an optional `run_id` param (auto-computed from the config+seed+fold when
  omitted, so even a bare `python train.py` produces addressable checkpoints).
- **Bootstrapped `tests/`**: `conftest.py` (synthetic tiny-dataset fixtures — a handful of random
  PNGs generated at test time, no dependency on the gitignored real `data/`),
  `test_orchestration.py` (16 tests: schema accept/reject, config_hash/run_id, manifest round-trip,
  ledger, sweep idempotency, and `test_determinism` — trains a real tiny UNet twice from the same
  seed on synthetic data and asserts bit-identical final weights with zero recorded
  non-determinism). Added `pytest>=8` to `requirements.txt` (not explicitly listed in Phase 0's
  dependency set, but obviously required to run the test suite this phase bootstraps).
- **Config layout relocation** (the "layout reconciliation" work, done as part of this phase since
  it's what the new schema validates against): `configs/experiment/base_config.yaml` →
  `configs/base.yaml`; new `configs/dataset/{clinicdb,colondb}.yaml`,
  `configs/model/{mkunet,gmkunet}/{t,s,base,m,l}.yaml`, `configs/training/emcad.yaml` fragments
  populate the previously-empty scaffold directories; all ~12 real experiment configs now compose
  these instead of duplicating full `model:`/`dataset:` blocks inline. `train.py`/`eval.py`
  (`--config`) and `search.py` (`--base-config`) CLI defaults now point at
  `configs/experiment/mkunet/mkunet_t_clinicdb.yaml` — `configs/base.yaml` alone is deliberately
  **not** a runnable standalone config anymore (see next item).
- **Bug found and fixed**: `configs/base.yaml` used to default `dataset.name`/`dataset.root` to
  ClinicDB. `configs/experiment/mkunet/mkunet_s_colondb.yaml` composed only the base config and
  never overrode either field — so despite its filename, its `experiment_name`, and its checkpoint
  directory (`checkpoints/mkunet_s_colondb/`, which contains 5 real trained fold checkpoints) all
  claiming ColonDB, **it was actually training and evaluating on ClinicDB the entire time**. Fixed
  by (a) making `dataset.name`/`root` required with no default in the schema — a config that
  forgets to select a dataset now fails to load instead of silently picking one — and (b) wiring
  `mkunet_s_colondb.yaml` to actually compose `configs/dataset/colondb.yaml`. Verified: the fixed
  config now loads 303 train / 38 val ColonDB pairs (vs. ClinicDB's 440/110). **The existing
  checkpoints under `checkpoints/mkunet_s_colondb/` predate this fix and are ClinicDB results —
  do not cite them as ColonDB results in the dissertation; re-run that config to get a real
  ColonDB checkpoint.**
- Also fixed while touching these files: all ~11 `gmkunet`/`mkunet` experiment configs had a
  `compose: ["../../base_config.yaml"]` path one directory level too deep (resolved to the
  nonexistent `configs/base_config.yaml`), pre-existing and unrelated to Phase 0/1 — every one of
  these configs was already unloadable before this session. `README.md`/`datasets/stats.py`
  updated to match the new paths.
- Verified end-to-end against the real pipeline throughout, not just unit tests: `eval.py` on
  `gmkunet_t_clinicdb` reproduces Phase 0's exact baseline numbers after all of the above
  (Dice 0.7816, mIoU 0.6881); a real `train.py` run confirms checkpoints now embed a correct
  `run_id`/`config_hash`/`git_commit`; `orchestration.runner.run_sweep()` verified against both a
  fake and the real `train.run_training`, including idempotent re-run skipping; `search.py` still
  completes a full grid sweep (summary CSV, best-config YAML, markdown report) with per-trial
  failure isolation intact.

## Phase 2 — Canonical metrics module (2026-08-28)

- **New top-level `metrics/` package**, collapsing the three independently-drifted Dice/IoU/HD95/
  ASD implementations (`utils/metrics.py`'s `get_binary_metrics`/`compute_dataset_metrics`,
  `utils/report.py`'s `compute_extended_metrics`, and `training/trainer.py`'s per-batch rolling
  proxy) down to two: this package (canonical) and the rolling proxy (deliberately kept separate,
  see below). `region.py` (dice/iou — empty-mask convention ported unchanged: both-empty → 1.0),
  `boundary.py` (hd95/asd/nsd — **behaviour change**: exactly-one-empty is now `None`/excluded +
  counted, replacing the old ad hoc `999.0` penalty that silently distorted any mean/percentile by
  however many empty-vs-nonempty pairs happened to land in a given eval; both-empty stays 0.0/1.0),
  `detection.py` (precision/recall/specificity/F2/accuracy — ported from `utils/report.py`
  unchanged, including its zero-denominator→0.0 convention; **note this doesn't match region.py's
  both-empty→1.0 convention** — pre-existing inconsistency, not introduced here, left as-is since
  Phase 2's scope was consolidation + the one documented HD95/ASD/NSD change, not a second
  unscoped convention change; adds `fpr_on_normals`/`specificity_on_lesion_free_subset`, both
  `None` — not 0.0 — when a dataset has no lesion-free images, which is true of ClinicDB/ColonDB),
  `calibration.py` (ECE, equal-mass binning — new, no prior implementation existed),
  `aggregate.py` (`EMPTY_MASK_CONVENTION` constant documenting every convention above in one place,
  `compute_dataset_metrics()`, `dice_p5`/`dice_p25`, `write_per_image_parquet()`).
- `utils/metrics.py` shrunk to just `count_parameters`/`measure_throughput`/`log_model_summary`
  (Phase 10's territory, not moved) — `get_binary_metrics`/`compute_dataset_metrics` deleted, not
  deprecated-and-reexported, since every call site was migrated in this same session rather than
  across a longer rollout. `utils/report.py`'s `compute_extended_metrics` deleted outright; its
  `EvaluationReporter.set_eval_results()` no longer takes `preds`/`gts` (no longer needed — the
  canonical `base_metrics` dict already carries precision/recall/specificity/f2/accuracy) and its
  "(N/A — multiclass)" table note is gone, since the new implementation computes those correctly
  for multiclass too (a genuine improvement: the old per-image TP/FP/FN/TN on a raw multiclass
  label map would have collapsed classes together, so it was skipped entirely before; the new
  version does a proper per-class one-vs-rest decomposition instead).
  `eval.py`/`training/trainer.py` now `from metrics import compute_dataset_metrics` directly
  instead of via `utils`'s re-export.
- Added `pyarrow==17.0.0` to `requirements.txt` for `write_per_image_parquet()` — the last release
  with a Python 3.8 wheel (18.0.0+ dropped 3.8; bump alongside a future Python bump).
- `training/trainer.py`'s rolling per-batch Dice/IoU (`train_one_epoch`, computed from raw logits,
  no medpy) is **unchanged** per the plan — it's a deliberately separate, cheap tensor-native
  computation for the progress bar/TensorBoard, not a second canonical implementation. Its
  docstring now says explicitly: never cite `train_dice`/`train_iou` in a results table.
  `validate()`'s metrics (the ones actually logged/checkpointed as `val_dice` etc.) already called
  the canonical function before this phase and still do, now pointed at the new package.
- **New `tests/test_metrics.py`** (7 tests): `test_metric_conventions` (every empty-mask rule in
  `EMPTY_MASK_CONVENTION`, checked against real function calls, not just read from the constant),
  exclusion-counting, the lesion-free-subset aggregates, an ECE sanity check, a Parquet
  round-trip, and `test_rolling_tracks_canonical_at_epoch_end` — builds a real `Trainer`, calls
  `train_one_epoch()` directly to capture the actual rolling `(dice, iou)` return value, then
  independently re-evaluates the same post-epoch weights on the same training data via the
  canonical path and asserts they land in the same ballpark (measured drift on the synthetic
  fixture: ~0.017; asserted tolerance: 0.35, generous on purpose to avoid flakiness while still
  catching a real convention-divergence bug). 23/23 tests passing repo-wide.
- Verified end-to-end against the real pipeline: `eval.py` on `gmkunet_t_clinicdb` still reproduces
  the exact baseline (Dice 0.7816, mIoU 0.6881, ..., all bit-identical to Phase 0/1) and its JSON
  report now carries `nsd`, `dice_p5`/`dice_p25`, `hd95_excluded_n`/`asd_excluded_n`, `ece`, and
  `fpr_on_normals`/`specificity_lesion_free` (`None` for ClinicDB, confirming the "no lesion-free
  images" case is detected correctly rather than silently reporting 0.0/1.0); a real multi-epoch
  `train.py` run confirms the canonical validation path (now backed by `metrics/`) still trains,
  checkpoints, and logs `val_dice`/`val_miou`/`val_hd95`/`val_asd` correctly.

## Phase 3 — Data-layer retrofit: (image, mask, meta) contract, leakage guards, BUSI/ISIC18 (2026-08-28)

- **`Dataset.__getitem__` now returns `(image, mask, meta)`** instead of `(image, mask)` —
  `meta = {subject_id, source_dataset, spacing, artefact_flags}`, applied project-wide (ClinicDB/
  ColonDB included, not just new datasets — a leakage guard is meaningless otherwise).
  `spacing` is `()`, not `None`, for "no spacing metadata" — confirmed torch 1.13's default
  `DataLoader` collate raises on a bare `None` inside a dict value but happily collates an empty
  sequence into `[]`. Single coordinated change across every consumer in one pass: found via a
  full-repo unpacking-pattern search, **5 call sites across 4 files**, not the ~4 named in the
  plan — `training/trainer.py` (`train_one_epoch`, `_validate_model`), `eval.py` (`evaluate`),
  `training/callbacks.py` (`PredictionOverlayCallback`), and **`utils/metrics.py`'s
  `measure_throughput`** (missed by an initial grep since it unpacks `(images, _)`, not
  `(images, masks)` — caught by actually running `eval.py --allow-test-eval` end-to-end afterward,
  not just by the grep). ClinicDB/ColonDB's `subject_id` honestly defaults to frame-level identity
  (filename stem) via each handler's new `ARTEFACT_FLAGS = {"frame_level_only_no_video_grouping":
  True}` — the caveat that used to live only in `KFoldDataModule`'s docstring now has a
  machine-readable home.
- **New `datasets/splits.py`**: `assert_no_subject_overlap()` (`LeakageError`),
  `load_published_split()` (reads + SHA1-hashes a `train_list_*.txt`-style file),
  `duplicate_cross_check()` (exact-hash cross-split duplicate detection — closes the gap in
  `_GenericHandler`'s unguarded random shuffle), plus `ExternalDatasetError` and
  `TestLoaderGuardError` (see below).
- **New `datasets/preprocess.py`**: `build_manifest()` (path/subject_id/split/`mask_empty`/
  resolution, computed once from the raw mask file — deliberately *not* inside `__getitem__`,
  where it would be both wasteful to recompute every epoch and ambiguous post-augmentation, since
  a crop can move a lesion in or out of frame run to run); `dedup()` (phash + SSIM, two-stage —
  mandatory for BUSI per spec).
- **Test-loader guard**: `StandardSplitDataModule.get_test_loader()` and
  `KFoldDataModule.get_test_loader()` now require a `token` argument, raising
  `TestLoaderGuardError` unless it was minted by the new
  `orchestration.ledger.LedgerWriter.issue_test_token()` (which appends a `Test_Evals` row as a
  side effect — "did this run touch the test set" is now a ledger fact, not a convention).
  `eval.py` gained `--test-token <TOKEN>` (pre-minted, e.g. from an orchestrated sweep) and
  `--allow-test-eval` (self-mints one for a manual run). Verified all four paths against real
  ClinicDB data: no token → refused with a clean error (not a raw traceback — added a
  `try/except TestLoaderGuardError` around the call, since a *bogus* token bypasses the earlier
  argparse-level check and needs its own handling); bogus token → refused; `--allow-test-eval` →
  works, ledger row recorded; pre-minted `--test-token` → works. `orchestration.runner`,
  `train.py`, `training/trainer.py`, and `search.py` contain zero references to `get_test_loader` —
  verified as a repo-wide static fact (`tests/test_data_contract.py::test_sweep_cannot_see_test`),
  so the sweep/training path structurally cannot reach the guarded test set.
- **`_GenericHandler` gains `external: bool`** (`dataset.external` in config): when set,
  `get_dataset("train")`/`get_dataset("val")` raise `ExternalDatasetError` instead of silently
  returning an empty loader — every sample goes to `"test"`. Registered handlers (ClinicDB et al.)
  don't support this; it's specifically for a held-out-only external cohort with no train/val split
  by design. Handler construction protocol changed to always pass `seed` (`DATASETS[name](cfg,
  seed)`, not `DATASETS[name](cfg)`) so a handler that needs to compute its own random split (BUSI)
  can, while ones that don't (ClinicDB/ColonDB/ISIC18 — published or official splits) just accept
  and ignore it.
- **New `datasets/busi.py`, `datasets/isic18.py`** — same `get_dataset()`/`get_kfold_pairs()`
  interface as ClinicDB/ColonDB, registered into `DATASETS`. **Structurally complete but UNVERIFIED
  against real data** — no BUSI or ISIC18 files exist anywhere reachable from this environment.
  Built against each dataset's publicly documented layout (BUSI: `benign/malignant/normal`
  subfolders, `<case>.png` + `<case>_mask.png`, no official split — hence a seeded random split via
  `dataset.split`, with mandatory `preprocess.dedup()`; ISIC18: the official
  `ISIC2018_Task1-2_*_Input`/`ISIC2018_Task1_*_GroundTruth` directory triad with a
  `<stem>_segmentation.png` mask suffix that needed its own pairing logic — the shared
  `MedicalSegmentationDataset` filename-matching only tolerates `_mask`/`_gt`). Verified the
  handler *interface* end-to-end against synthetic data built to match each documented layout
  (`tests/test_data_contract.py`) — get_dataset/get_kfold_pairs/mask-pairing/dedup all confirmed
  working structurally; **sanity-check directory names and the mask suffix against your actual
  download before first real use.** New `configs/dataset/busi.yaml`/`isic18.yaml` fragments, same
  "point this at real data, unverified" caveat.
- **New `tests/test_data_contract.py`** (13 tests): the four the plan names —
  `test_no_subject_overlap` (including a real check against ClinicDB's actual K-Fold split, not
  just synthetic ids), `test_external_never_trained`, `test_test_loader_guard`,
  `test_sweep_cannot_see_test` — plus contract round-trip tests (including through a real
  `DataLoader`, where the `None`-vs-`()` collate issue above actually surfaces),
  `duplicate_cross_check` unit tests, and the BUSI/ISIC18 handler-interface tests. 36/36 tests
  passing repo-wide.
- **Real finding**: ran the new `duplicate_cross_check()` against ColonDB's actual published
  split (not just as a unit-test exercise) and it found **35 of ColonDB's 379 images (9.2%) are
  byte-identical duplicates split across train/val/test** — 9 duplicate groups, most crossing the
  train/test boundary specifically (confirmed genuine: matching MD5 + file size + real image
  content, not blank/degenerate frames — e.g. `train/images/198.png` and `val/images/183.png` are
  byte-identical). This means any model trained on ColonDB's current published split (including
  the newly-fixed `mkunet_s_colondb` config from Phase 1) has some fraction of its "held-out" test
  images already memorisable from training. **Not fixed here** — per user instruction, no
  checkpoints were re-run this session, and de-duplicating a published split is a data-curation
  decision the user should make deliberately (e.g. via `datasets.preprocess.dedup()` or
  `datasets.splits.duplicate_cross_check()`-guided exclusion), not one this pass makes silently.
  Ran the same check against ClinicDB: 0 cross-split duplicates found.
- `MedicalSegmentationDataset.__init__` gained `source_dataset`/`subject_id_fn`/`artefact_flags`
  params (all optional, default to `""`/frame-level-stem/`{}`); `KFoldDataModule.get_fold_loaders()`
  now threads `source_dataset`/`artefact_flags` through explicitly (it constructs
  `MedicalSegmentationDataset` directly from cached fold pairs, bypassing the handler's
  `get_dataset()`, so couldn't inherit them the way `StandardSplitDataModule` does).

## Phase 4 — Channel construction module + modality-driven augmentation policy (2026-08-28)

- **New `datasets/channels.py`**: `build_channels(image, mode, order)` for m1 (rgb) / m2 (rgb+xy) /
  m3 (rgb+ycbcr) / m4 (rgb+xy+rtheta) / m5 (all), built from named channel *groups* (not hardcoded
  per-mode logic) so a caller can pin an explicit `order` — the group boundaries this produces are
  what Phase 11's per-group Shapley attribution will slice by. `xy_channels()` ([-1,1] position),
  `r_theta_channels()` (radius + sin/cos(θ) — sin/cos specifically to avoid the -π/π branch cut a
  raw angle channel would have along the negative x-axis), `randproj_channels()`/
  `coordonly_channels()` ablation controls, `modality_effective_channels()` (drops the `ycbcr`
  group for `modality=grayscale` — closes `models/baseline/emcad.py`'s old ad hoc
  "if grayscale, repeat to 3ch" gap at the *data* level instead of patching every model), per-mode
  channel-stats caching (`channel_norm_stats()`, hashed by dataset+mode+modality).
  `models/proposed/gmk_unet.py`'s private `_ycbcr()` now imports `ycbcr_from_rgb_tensor()` from
  here instead of defining its own — same BT.601 coefficients (`_YCBCR_COEFFS`), shared with the
  numpy path `build_channels()` uses, so the two can never numerically drift apart even though one
  runs on live GPU tensors during a forward pass and the other on numpy arrays at data-loading
  time (confirmed 0.0 max diff between the two on a real ClinicDB image). Verified the real
  `gmkunet_t_clinicdb` checkpoint still reproduces the exact baseline (Dice 0.7816) after this
  swap.
- **New `datasets/augment.py`**: `AugmentationPolicy`, built once per dataset from
  `(modality, ds_cfg)`. Enforces ORDER=post (geometric augmentation via Albumentations first, then
  `build_channels()` regenerates XY/Rθ from the *already-augmented* frame — a pre-augmentation XY
  channel would encode stale coordinates once the frame is flipped/rotated). Modality-conditioned
  augmentation intensity (spec §5.4's colour vs. grayscale-ultrasound vs. grayscale-microscopy):
  the config schema only exposes the coarse `dataset.modality: colour|grayscale` the spec fixes at
  dataset level, so the finer ultrasound-vs-microscopy split (needed only for augmentation
  intensity, not channel construction) is resolved from `dataset.name` via a small
  `_GRAYSCALE_SUBTYPE` table (`{"busi": "ultrasound"}` today; no microscopy dataset registered
  yet).
- Config schema gained `dataset.modality` (`colour|grayscale`, default `colour`), `channel_mode`
  (`m1`..`m5`, default `m1`), `channel_order`. All three are inert by default — every existing
  config stays on `m1`/`colour` (today's plain-RGB pipeline, byte-for-byte unchanged) unless a
  config explicitly opts into a different mode.
- **Deliberately not wired into the live training pipeline this phase**: `datasets/dataset.py`/
  `datasets/datamodule.py` still call `transforms.build_transforms()` directly, not
  `AugmentationPolicy` — no model in Phases 0–5 accepts non-RGB input, so there's nothing yet to
  consume `channel_mode` > m1. Phase 6 is explicitly where this gets wired in for real: the Mamba
  model's auxiliary VSS encoder is specified to consume the same multi-channel input as the
  primary encoder, via `dataset.channel_mode`.
- **New `tests/test_channels.py`** (9 tests): the five the plan names —
  `test_val_test_no_augment`, `test_mask_interpolation`, `test_channel_order`,
  `test_theta_continuity`, `test_grayscale_drops_colour` — plus a BT.601 reference-value check and
  an `AugmentationPolicy` dataset-specific intensity-scale check. Found and fixed two real bugs
  while writing these: the first `test_theta_continuity` draft swept a row that passed through the
  coordinate origin itself (a genuine r=0 singularity no encoding can smooth over, different from
  the branch-cut discontinuity sin/cos is actually meant to fix) rather than testing the branch cut
  itself — rewrote it to sweep a fixed negative-x *column* through y=0, which is where the branch
  cut actually lives, and added a direct comparison confirming a raw `atan2` channel really does
  jump there (~2π) while sin/cos doesn't. Also fixed a wrong transform-list index (`[-2]` instead
  of `[-3]`) in the augmentation-intensity test. 45/45 tests passing repo-wide.

## Phase 5 — Model registry hardening: capacity control, budget guard, parameter-group optimizer (2026-08-28)

- **New `models/build.py`**: `build_width_matched(base_model_fn, width_candidates, input_shape,
  target_params, target_flops, tol)` — searches a model family's already-exposed width-scaling
  knob (MK-UNet/GMK-UNet's channels-list presets) for a configuration matching a target
  param/FLOPs budget. Verified against real MK-UNet presets: finds an exact match (0.0 relative
  error) when the target is in the candidate list, correctly reports `within_tolerance: False`
  with the closest candidate otherwise.
- **`models/registry.py`**: `ModelRegistry.get()` (and `get_model()`) gained
  `budget_ceiling`/`allow_over_budget` — profiles every built model immediately and raises
  `ModelBudgetExceededError` unless the override is explicit. Applies to every registered model,
  including whichever one Phase 6 adds — the `@register` decorator API itself is unchanged.
- **`training/optimizers.py`**: `build_optimizer(cfg, model)` — signature changed from
  `build_optimizer(cfg, params)` (confirmed zero param-group splitting before this) to take the
  whole model, not `model.parameters()`, since the new `no_decay_group(model)` needs each
  parameter's dotted name and owning module to split by *name* (bias / normalisation-layer
  affine params / — forward-looking for Phase 6 — a parameter literally named `A_log` or `D`),
  not by tensor shape. Two param groups (`weight_decay` / `0.0`) built and logged for every
  optimizer choice (Adam/AdamW/SGD). Updated both call sites: `train.py`'s main path and
  `training/trainer.py`'s multi-stage `_fit_staged()` rebuild (which used to pass a
  `filter(requires_grad, ...)` iterator — `no_decay_group()` already skips frozen params via its
  own `requires_grad` check, so passing the whole model post-`_set_frozen()` produces the same
  trainable-only set without a separate filter step).
- **Real finding, fixed**: this param-group change breaks resuming from any checkpoint saved
  under the old single-group optimizer (`optimizer.load_state_dict()` raises `ValueError` on a
  param-group-count mismatch) — caught running the real training smoke test, not a unit test.
  Fixed in `utils/checkpoint.py`'s `CheckpointManager.load()`: catches that specific `ValueError`,
  logs a clear warning, and continues with a freshly-initialised optimizer rather than crashing
  the whole resume — model weights (the expensive part) still load and restore correctly; only
  Adam's per-parameter moment estimates are lost, once, for any checkpoint that predates this
  change. Verified end-to-end: a real resume from a pre-existing checkpoint now degrades
  gracefully with a warning instead of crashing, and training continues correctly from there.
- **New `tests/test_optim.py`** (4 tests) and **`tests/test_models.py`** (6 tests): the two the
  plan names — `test_param_groups`, `test_capacity_control_match` — plus frozen-param exclusion,
  SGD param-grouping, no-target/no-candidates validation, and the registry budget guard
  (raises/bypasses/not-triggered). 56/56 tests passing repo-wide.

## Phase 5.5 — MVP checkpoint reached (2026-08-28)

Verified per the plan's own criterion: ran `orchestration.runner.run_sweep()` across 2 real model
families (`mk_unet_t`, `gmk_unet_t`) × 2 seeds on real ClinicDB data — 4 real training runs, not
mocked. Confirmed: distinct `config_hash` per family (shared across seeds), distinct `run_id` per
(family, seed), correct manifests (resolved config, git commit + dirty flag, env hash, hardware,
timing) and ledger rows written for all 4; re-running the identical sweep call afterward correctly
reports all 4 as `skipped-done` with zero retraining. Along the way, found that the two seeds'
validation Dice was suspicious identical — investigated directly (not dismissed): confirmed via a
3-seed direct comparison that *training* metrics (loss, train Dice) do differ correctly per seed,
but a ~40K-parameter model after a single epoch collapses to an identical hard-thresholded
all-background validation prediction regardless of seed, which makes Dice/HD95/ASD — computed on
the thresholded output, not the raw logits — genuinely identical by construction, not a
determinism bug. (Also incidentally confirmed Phase 2's boundary-metric exclusion behaviour is
working as designed: an all-background prediction against ClinicDB's always-non-empty ground
truth excludes every sample from HD95/ASD, correctly producing 0.0 as "mean of zero excluded
samples," not a misleadingly "perfect" score — visible via `hd95_excluded_n` for anyone who checks
it, which is the point of tracking that field at all.)

Phases 0–5 now let a user run config-hashed, leakage-guarded, canonically-metriced experiments
across all 6 existing model families (5 baselines + GMK-UNet ×5 sizes) on ClinicDB/ColonDB (BUSI/
ISIC18 handlers are in place but unverified against real data — see Phase 3's entry), with a
working idempotent sweep/ledger. Per the plan, Phase 6 (Mamba) is dissertation-critical but not
MVP-blocking.
