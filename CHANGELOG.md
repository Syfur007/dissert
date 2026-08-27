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
