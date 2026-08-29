# dissert

**A config-driven PyTorch framework for medical image segmentation research**

> One YAML fully determines a run. Baseline and proposed architectures are trained, evaluated, and
> statistically validated under one shared pipeline — training, evaluation, statistics,
> attribution, robustness, uncertainty, profiling, and reporting are each a first-class module, not
> a notebook. The full contract is [`Technical_Framework_Spec.md`](Technical_Framework_Spec.md); if
> code and that document disagree, one of them is a bug.

## 1 · QUICKSTART

| Step | Command |
| --- | --- |
| Install | `conda activate thesis && pip install -r requirements.txt` |
| Test | `pytest tests/ -v` — 310 tests |
| Train | `python train.py --config configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml` |
| Evaluate | `python eval.py --config <same config> --fold 0 --allow-test-eval` |
| Reproduce | `./scripts/reproduce.sh` — a real, reduced end-to-end pass |

CUDA build and the Mamba fused-kernel dependency notes live in `requirements.txt`'s comment block.

## 2 · LAYOUT

| Layer | Package(s) | Purpose |
| --- | --- | --- |
| Config | `configs/`, `orchestration/schema.py` | `compose:`-merged YAML, schema-validated on load |
| Data | `datasets/` | Loaders, channel construction, augmentation, leakage guards |
| Models | `models/` | UNet family, EMCAD, GMK-UNet, Mamba/VSS hybrid — one registry |
| Training | `training/`, `losses/` | Trainer, optimizer, determinism, declarative losses |
| Metrics | `metrics/` | The one canonical Dice / IoU / HD95 / ASD / NSD / ECE source |
| Orchestration | `orchestration/` | Run manifest, ledger, sweeps |
| Analysis | `stats/` `profiling/` `attribution/` `uncertainty/` `robustness/` `analysis/` | Significance testing, efficiency, explainability, robustness |
| Reporting | `reporting/` | Manuscript tables/figures, blocking rules |

## 3 · MODELS

| Registry name | Kind |
| --- | --- |
| `unet`, `attention_unet` | Baselines |
| `mk_unet` (+ `_s` / `_t`), `emcad` | Baselines |
| `gmk_unet` | Proposed — geometry/colour channel groups, gated fusion |
| `mamba_unet` | Proposed — dual-branch MK-UNet + VSS/selective-scan hybrid |

## 4 · CHANNEL MODES

| Mode | Groups |
| --- | --- |
| m1 | RGB |
| m2 | RGB + XY |
| m3 | RGB + YCbCr |
| m4 | RGB + XY + Rθ |
| m5 | RGB + XY + YCbCr + Rθ |

## 5 · GUARANTEES

| Guarantee | Enforced by |
| --- | --- |
| Test set touched once, on purpose | `datasets.datamodule.get_test_loader(token)` raises without a minted token |
| Every run is addressable | `run_id = R-{config_hash[:7]}-s{seed}-f{fold}`, in every manifest/checkpoint/artefact |
| No config-drift between models | Shared `AugmentationPolicy`; no per-model augmentation key in the schema |
| Reported tables are trustworthy | `reporting/` refuses a dirty-tree run, an under-seeded config, an unstated comparison, or unsanitised saliency |

## 6 · DOCUMENTS

| Document | Contents |
| --- | --- |
| [`Technical_Framework_Spec.md`](Technical_Framework_Spec.md) | The specification — every module, contract, and invariant |
| [`CODE_REVIEW.md`](CODE_REVIEW.md) | Deep implementation reference — file-by-file, function-by-function |
| [`CHANGELOG.md`](CHANGELOG.md) | Build history, phase by phase |
