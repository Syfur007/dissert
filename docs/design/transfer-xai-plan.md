# Plan: Repository Reorganisation, Transfer Learning, Pretrained Weights, and Explainable AI

**Status:** proposal, not started · **Date:** 2026-09-25 (rev. 2) · **Owner:** Syfur007
**Moves to:** `docs/design/transfer-xai-plan.md` as part of Phase 0.

> **Platform constraint (set by the owner, rev. 2): Python ≤ 3.10 and torch 1.13.x.**
> The whole plan now targets **Python 3.10 + torch 1.13.1 / torchvision 0.14.1**. Every library
> and encoder named below was installed and exercised in a throwaway Python 3.10.21 + torch 1.13.1
> (CPU) environment on 2026-09-25; §1.4 has the results. Anything that needs torch 2.x (SAM2,
> MedSAM2, captum ≥0.9, current peft/transformers, zennit 1.0, MONAI 1.6) is out of scope, not
> deferred.

This document has three parts:

- **Part A (§1–§3):** what the repo has today, and what current research and tooling support.
  Library versions, weight availability and licences in §3 were checked against PyPI and the
  Hugging Face Hub on 2026-09-25.
- **Part B (§4–§9):** the plan. Phase 0 reorganises the repo and **ships before any new feature**.
  Phase 1 moves to Python 3.10 on the same torch 1.13.1, which the new libraries need. Phases 2–5
  add the features.
- **Part C (§10–§12):** risks, decisions that need the owner, and references.

---

## 0 · Summary

| # | Phase | Changes behaviour? | Gate to finish |
| --- | --- | --- | --- |
| 0 | **Reorganisation:** `src/` layout, a single `dissert` package, `pyproject.toml`, docs moved to `docs/`. The repo root goes from 20 tracked directories to 7. | No. Files move and imports are rewritten, nothing else. | Same test count passes. Every `config_hash` is unchanged. A pre-move checkpoint still loads. |
| 1 | **Platform move:** Python 3.8 → 3.10. **torch stays 1.13.1**, and numpy, scipy, scikit-learn, scikit-image and pandas stay pinned. timm 0.6.12 → 1.0.30, opencv 5.0 → 4.11 (forced), and unused dependencies are removed. | Model initialisation and forward pass: **no** (bit-identical, verified). Training augmentation: **yes, slightly** (opencv's `warpAffine` changed). | Already met in a dry run: 357 passed / 2 skipped on 3.10, the same as on 3.8. Remaining: the reporting rule that stops mixing environments, and GPU runs on the new stack. |
| 2 | **Pretrained-weight infrastructure:** one encoder factory, recorded weight provenance, failure on missing weights, input-stem adaptation for 5–11-channel inputs, pretrained normalisation. | Yes | Parity test against the current EMCAD/PVT path. Weight identity is part of `config_hash`. |
| 3 | **Transfer-learning strategies:** discriminative LR, LP-FT and gradual unfreezing (built on the existing `stages`), BN freezing, `init_from` for cross-dataset fine-tuning, optional LoRA. | Yes | Unit tests for parameter groups, freezing and BN. |
| 4 | **XAI:** standard libraries (Captum, pytorch-grad-cam, Quantus) behind one segmentation-target abstraction, faithfulness and randomisation metrics, and a `dissert-explain` CLI that writes into the run directory. | Adds features only | Library-vs-own-implementation cross-checks. The reporting rule consumes the new outputs. |
| 5 | **Study protocol and reporting:** transfer and XAI experiment grids, new tables and figures, new blocking rules. | Adds features only | Report renders from artefacts alone. |

Deferred until there is evidence they are needed (§9): in-domain self-supervised pretraining,
LoRA, concept-based XAI.

**Excluded by the torch 1.13 constraint:** SAM2 and SAM 2.1 (need torch ≥2.5.1), MedSAM2 (built on
SAM2), captum ≥0.9 (torch ≥2.3), current peft (pulls transformers/accelerate releases that need
torch ≥2), zennit 1.0 (Python ≥3.11.11), and MONAI 1.6 (torch ≥2.8).

---

# Part A — Findings

## 1 · Current state of the repo (audit)

### 1.1 Structure

- **20 tracked root directories.** Sixteen of them are top-level Python packages: `analysis`,
  `attribution`, `datasets`, `losses`, `metrics`, `models`, `orchestration`, `profiling`,
  `reporting`, `robustness`, `stats`, `training`, `uncertainty` and `utils` (plus `configs`,
  `env`, `notebooks`, `scripts`, `tests` and `.github`). None of these packages is installed:
  imports work only through `pytest.ini`'s `pythonpath = .` and `scripts/reproduce.sh`'s
  `export PYTHONPATH`.
- **Generic top-level package names clash with PyPI distributions.** The worst case is
  `datasets`, which shadows Hugging Face `datasets`, and the HF stack that the new features pull
  in can import it. `stats`, `metrics`, `models`, `utils`, `profiling` and `analysis` carry the
  same risk. Putting everything under a `dissert.` namespace removes the whole class of problem.
- **`utils/` mixes unrelated code:** config loading, checkpoint I/O, early stopping, parameter
  counting, a 750-line evaluation reporter, plotting and logging.
- **The root holds five planning documents** (`CODE_REVIEW.md`, `OUTPUT_LAYOUT.md`,
  `SESSION_GROUPING_PLAN.md`, untracked `XDASH_RESUME_CONTRACT.md`, `CHANGELOG.md`).
  `CODE_REVIEW.md` links to `Technical_Framework_Spec.md` and `IMPLEMENTATION_PLAN.md`, and
  neither file is in the repo.
- **Stray runtime directories:** a root `artifacts/`, created by `datasets/channels.py`'s
  `cache_dir="artifacts/channel_stats"` default, and a legacy pre-hash layout under
  `outputs/{checkpoints,logs,runs,artifacts}`.
- **Declared but never imported** (checked with `git grep`): `warmup-scheduler`, `transformers`,
  `torchprofile`, `torchmetrics`, `einops`, `ptflops`, `torchinfo`, `torchsummary`,
  `torchsummaryx`, `segmentation-mask-overlay`, `tifffile`, `SimpleITK`, `nibabel`, `h5py` and
  `ml_collections`. Kaggle installs every one of them in every session.

### 1.2 Pretrained weights and transfer learning

| Area | Today | Problem |
| --- | --- | --- |
| Which models use pretrained weights | EMCAD only: a PVTv2 encoder via `models/backbones.py` | MK-UNet, GMK-UNet and Mamba-UNet always train from scratch. |
| Weight loading | `PVT_Wrapper` reads `./pretrained_pth/pvt/<name>.pth` | **If the file is missing it prints a warning and silently uses random weights.** The config still says `pretrain: True` and the `config_hash` is identical, so a random-init run cannot be told apart from a pretrained one. |
| Key matching | `{k: v for k in save_model if k in model_dict}` | Missing or unexpected keys are dropped without any report, so partial loads pass silently. |
| timm fallback | `except Exception` falls back from timm to the local ResNet | A timm error silently switches the implementation and the weight source. |
| Input channels | EMCAD accepts 1 or 3 channels (a 1→3 conv) | Channel modes m2–m5 (5–11 channels) cannot use any pretrained stem. |
| Normalisation | Per-dataset `norm_mean/std` | Pretrained encoders expect the statistics of their pretraining data. Nothing reconciles the two. |
| Freezing | `stages[].freeze`, substring match on module names | Covers basic LP-FT and gradual unfreezing. There are no per-group learning rates, no BN-statistics freezing and no layer-wise decay. |
| Capacity guard | `count_parameters()` counts trainable parameters only | A frozen 300M-parameter encoder passes `budget_ceiling`. Total parameters must be reported as well. |
| Library versions | `timm==0.6.12`, `huggingface-hub==0.11.0` | Too old to load current Hub weights (DINOv3, SAM-ViT, Hiera, ConvNeXt-V2 FCMAE). |

### 1.3 Explainability

`attribution/` is already good. It has:

- channel-group occlusion and exact Shapley over channel groups;
- Integrated Gradients, aggregated per channel;
- our own Seg-Grad-CAM and Seg-XRes-CAM;
- Mamba auxiliary-branch ablation and a CBFFM gate probe;
- a cascading model-parameter randomisation check (SSIM) and a label-randomisation check;
- a reporting rule that refuses unsanitised saliency.

Gaps:

1. **No faithfulness metrics** (deletion/insertion, ROAD, region perturbation). Today a heatmap
   can pass the sanity check without anyone measuring whether it reflects what the model uses.
2. **No pixel-level gradient maps** (IG, SmoothGrad, GradientSHAP) and no model-agnostic
   perturbation maps (RISE, superpixel ablation).
3. **Only the top-down MPRT sanity check**, which has documented shortcomings (§2.2.4).
4. **No CLI.** Everything is library-only (`scripts/reproduce.sh` prints the functions to call),
   so nothing lands in the run directory by default.
5. **No ViT or Swin support** (no `reshape_transform`), which Phase 2 encoders will need.

### 1.4 Platform

The environment is Python 3.8.20, torch 1.13.1, timm 0.6.12 and numpy 1.22.4. **Python 3.8
reached end of life in October 2024.** Both Kaggle notebooks build a Python 3.8 venv from the
deadsnakes PPA in every session.

The owner's ceiling is **Python 3.10 + torch 1.13.x**. Everything in the table below was
**installed together and run** in a Python 3.10.21 + torch 1.13.1 (CPU) environment, with the
repo's numeric pins kept (numpy 1.22.4, scipy 1.10.1, scikit-learn 1.3.2, scikit-image 0.21.0,
pandas 2.0.3). `pip check` reported no conflicts.

| Library | Version to pin | Result on Python 3.10 + torch 1.13.1 |
| --- | --- | --- |
| numpy | **1.22.4 (must stay <2)** | Required. With numpy 2.2.6, importing torch 1.13 fails ("A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"). |
| timm | 1.0.30 | Works. timm's own CI tests down to "PyTorch 1.13 + Python 3.10" on its lower end. Verified `features_only` forward and backward for `pvt_v2_b2`, `pvt_v2_b0.in1k` (pretrained download), `resnet50` (`in_chans=8`), `convnextv2_atto.fcmae` (pretrained, `in_chans=1`), `efficientnet_b0`, `hiera_tiny_224`, `mambaout_femto`, `vit_small_patch16_dinov3` and `samvit_base_patch16` (`img_size=256`). `adapt_input_conv`, `freeze_batch_norm_2d` and `param_groups_layer_decay` all present. |
| huggingface-hub | current (2.0.0 needs Python ≥3.10) | Works; it's what timm uses to download weights. |
| captum | **0.8.0** (the last release allowing torch <2.3) | Works. IntegratedGradients, NoiseTunnel (SmoothGrad), GradientShap, LayerGradCam and FeatureAblation all ran. |
| grad-cam (pytorch-grad-cam) | 1.5.7 | Works. GradCAM, HiResCAM, GradCAM++, LayerCAM, ScoreCAM, AblationCAM and EigenCAM ran with `SemanticSegmentationTarget`. `SegEigenCAM` is exported. The ROAD metrics (`ROADMostRelevantFirst/LeastRelevantFirst/Combined`) ran on a segmentation target. |
| quantus | 0.6.0 | **Partly works.** RegionPerturbation, PixelFlipping, MaxSensitivity, Sparseness, Complexity, RelevanceMassAccuracy, PointingGame, MPRT, SmoothMPRT and EfficientMPRT ran. **FaithfulnessCorrelation fails**: on Python ≥3.10, Quantus calls `scipy.stats.pearsonr(..., axis=1)`, which needs scipy ≥1.14, and scipy 1.14 needs numpy ≥1.23.5. Quantus's **ROAD and IROF** also failed in the smoke test (index and shape errors); not investigated further, because grad-cam's ROAD works. |
| segmentation-models-pytorch | 0.5.0 | Works (`Unet("tu-convnextv2_atto", in_channels=5)`). |
| mamba-ssm / causal-conv1d | 1.0.1 / 1.1.1 (the current pins) | GitHub-release wheels exist for `cu118torch1.13…cp310`, so only the `cp38` tag in the install URL changes. |
| **Out of reach** | n/a | SAM2 (torch ≥2.5.1), captum 0.9 (torch ≥2.3), current peft (its accelerate dependency needs torch ≥2.0 from 1.0 onward, and transformers declares torch ≥2.0 from 4.49 onward and ≥2.5 in 5.x; an old peft/transformers/accelerate trio pinned for torch 1.13 was not tested), zennit 1.0 (Python ≥3.11.11), MONAI 1.6 (torch ≥2.8), mamba-ssm 2.3 (triton ≥3.5). |

Two consequences of staying on torch 1.13:

- **No fused attention.** `scaled_dot_product_attention` only arrived in torch 2.0. timm falls back
  to plain attention, so ViT encoders (DINOv3, SAM-ViT) run slower and use more memory than
  published numbers suggest.
- **Frozen ecosystem.** New libraries increasingly require torch ≥2. Every future dependency must
  pass the same Python 3.10 + torch 1.13 smoke test before adoption (§10).

---

## 2 · What current research and tooling support

### 2.1 Transfer learning and pretrained weights

#### 2.1.1 Where to get weights (availability checked on the HF Hub, 2026-09-25)

| Source family | Concrete weights | Licence | Evidence for medical segmentation |
| --- | --- | --- | --- |
| Supervised ImageNet (CNN and hierarchical transformer) | `timm/pvt_v2_b0…b5.in1k`, ResNet, EfficientNet and ConvNeXt `*.in1k` | Apache-2.0 (timm) | The default baseline. EMCAD and PraNet-lineage polyp models use a PVTv2-b2 encoder. |
| Self-supervised, natural images (all verified to build and backprop on torch 1.13 via timm 1.0.30) | ConvNeXt-V2 `*.fcmae`, Hiera `*.mae`, DINOv3 `timm/vit_{small,base}_patch16_dinov3.lvd1689m` | FCMAE and MAE Apache/CC-BY-NC (check each card); **DINOv3 uses Meta's custom "DINOv3 License"** (ungated on timm, gated on `facebook/`) | Top SSL ImageNet models transfer better to medical tasks than supervised ImageNet [T3]. For polyp generalisation, **SSL ImageNet (MoCo v3, ResNet-50) generalised best Kvasir→ClinicDB, beating in-domain HyperKvasir pretraining; ViT-B scored higher in-domain but generalised worse** [T6]. |
| DINOv3 as a frozen dense encoder | DINOv3 ViT-S/B/L | DINOv3 License | Frozen DINOv3 with a light readout reaches 0.895 Dice on Kvasir-SEG and 0.897 on ISIC 2018 [T9]. Dino U-Net uses a frozen DINOv3 with an adapter [T10]. **MedDINOv3 finds that plain ViTs still lag strong CNN baselines on medical dense prediction** without multi-scale token aggregation and domain pretraining [T11]. |
| Segmentation foundation models | SAM ViT `timm/samvit_*_patch16.sa1b` (Apache-2.0; **verified on torch 1.13** at `img_size=256`). SAM 2.1 Hiera `facebook/sam2.1-hiera-*` and MedSAM2 `wanglab/MedSAM2` are **excluded: SAM2 needs torch ≥2.5.1.** | as listed | Fine-tuned SAM2 encoders report strong polyp results, for example about 0.948 mDice on ClinicDB and 0.918 Kvasir→ClinicDB in one study [T13], but that route is closed under the platform constraint. Parameter-efficient adaptation (LoRA in SAMed, adapters in Med-SA and SAM2-UNet) is the norm at this scale [T14]. |
| Radiology-specific | RadImageNet (1.35M CT/MRI/US images) | check the release terms | Beats ImageNet on several radiology tasks, but **not uniformly**: ImageNet won most small-target classification tasks in one comparison [T4, T5]. Relevant to BUSI (ultrasound), not to endoscopy or dermoscopy. |
| Endoscopy-specific | EndoDINO (ViT-B/L/g, up to 10M images) [T7], GastroNet-5M-pretrained models [T8], Endo-FM (video) [T12], EndoViT | **Public weights not found on the HF Hub under these names.** Treat as "if obtainable". | EndoDINO as a frozen encoder with simple heads reached SOTA polyp segmentation. The GastroNet-5M model beat the benchmarks on lesion delineation, including polyps. |
| Biomedical vision-language | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (MIT) | MIT | A global-image encoder, a weak fit for dense segmentation. Low priority. |
| Our own checkpoints | `best.pth` from a source dataset | n/a | The standard polyp generalisation protocol trains on Kvasir-SEG + ClinicDB and tests on ColonDB, ETIS and CVC-300 [T15]. This repo already has ClinicDB and ColonDB. |

What this evidence adds up to:

1. **Pretraining matters most when labelled data is small.** ClinicDB has 612 images. With large
   targets and long schedules, from-scratch training catches up [T1, T2]; this repo's datasets
   are small.
2. **In-domain is not automatically better.** Treat the pretraining source as an experimental
   factor and compare several; don't pick one by assumption [T3, T5, T6].
3. **ViT foundation encoders need multi-scale necks and can overfit on small sets** [T6, T11].
   Hierarchical encoders (PVTv2, ConvNeXt, Hiera) drop into U-Net-style decoders directly.

#### 2.1.2 Fine-tuning strategies (standard and citable)

| Strategy | Mechanism | Source |
| --- | --- | --- |
| Full fine-tuning | All weights trainable, single LR | baseline |
| Frozen encoder / linear probe | Train decoder or head only | [T16] |
| **LP-FT** | Train the head with the encoder frozen, then fine-tune everything at a lower LR. Plain fine-tuning with a random head distorts pretrained features. LP-FT gave about +1% ID and about +10% OOD over full fine-tuning in its benchmark. | Kumar et al., ICLR 2022 [T16] |
| Discriminative LR / layer-wise LR decay | The LR shrinks with depth toward the input (decay about 0.65–0.75 is standard in BEiT/MAE ViT recipes) | ULMFiT [T17]; BEiT and MAE fine-tuning recipes |
| Gradual unfreezing | Unfreeze from the top down, stage by stage | ULMFiT [T17] |
| Surgical fine-tuning | Tune only the block group that matches the kind of shift (input-level shift → early layers) | Lee et al., ICLR 2023 [T18] |
| L2-SP | Penalise ‖θ − θ_pretrained‖² instead of plain weight decay | Li et al., ICML 2018 [T19] |
| BN handling | Keep BN running statistics frozen in frozen stages and at small batch sizes (batch 8 here) | common practice (timm `freeze_batch_norm_2d`) |
| PEFT (LoRA, adapters) | Low-rank or bottleneck updates on a frozen ViT. On torch 1.13 the `peft` library is effectively unavailable (§1.4); a LoRA wrapper for `nn.Linear` is about 30 lines. | LoRA [T20]; SAMed and Med-SA [T14] |

#### 2.1.3 More than 3 input channels (specific to this repo)

Channel modes m2–m5 add XY, YCbCr and Rθ channels, while every pretrained stem is RGB.

- timm's `adapt_input_conv` handles `in_chans > 3` by **tiling the RGB kernels and rescaling them
  by 3/in_chans** [T21]. That treats XY or Rθ as more colour, which is semantically wrong, and it
  changes what the stem computes on the RGB part at initialisation.
- **Recommended: `zero_init`.** Copy the pretrained RGB kernels into the RGB slots and zero the
  kernels of the extra channels. At step 0 the network computes exactly what the pretrained model
  computes on RGB, and the extra channels are learned from zero. This is the same
  function-preserving idea used when input convolutions of pretrained generative models are
  widened.
- **Alternative: `separate_stem`.** A small from-scratch stem for the non-RGB groups, added to
  the pretrained stem's output. This keeps the geometry pathway explicit, which is useful for the
  existing channel-group attribution.
- Offer `repeat_scale` (timm's behaviour) only as an ablation arm.

#### 2.1.4 Normalisation

Pretrained encoders publish their statistics (`model.pretrained_cfg["mean"/"std"]` in timm).
**Rule:** when an RGB pretrained encoder is used, normalise the RGB channels with the encoder's
statistics. Non-RGB channels keep the repo's existing handling. Record the statistics actually
used in the manifest.

### 2.2 Explainable AI for segmentation

#### 2.2.1 Framing the target

A segmentation model has no single logit to explain. The standard move, from Seg-Grad-CAM [X1],
is to explain a scalar: **the sum of logits (or probabilities) over a pixel set of interest**,
such as the predicted mask, the ground-truth mask, a user ROI, or a boundary band. Every method
below plugs into that one target definition, so it should be implemented exactly once.

#### 2.2.2 Methods

| Family | Methods | Standard implementation | In repo? |
| --- | --- | --- | --- |
| Gradient-weighted CAM | Seg-Grad-CAM [X1]; Seg-XRes-CAM [X2]; Seg-HiRes-Grad CAM [X3]; HiResCAM, Grad-CAM++, LayerCAM | **pytorch-grad-cam** (`grad-cam` 1.5.7, verified on torch 1.13) with `SemanticSegmentationTarget`; ViT and Swin via `reshape_transform` [X4] | Seg-Grad-CAM and Seg-XRes-CAM (own implementation) |
| Gradient-free CAM | ScoreCAM, AblationCAM, EigenCAM, SegEigenCAM (segmentation-specific; exported by grad-cam 1.5.7) | pytorch-grad-cam [X4] | No |
| Path and gradient attribution | Integrated Gradients, SmoothGrad (NoiseTunnel), GradientSHAP, DeepLIFT | **Captum 0.8.0**, the last release that allows torch <2.3 (verified) [X5]; Captum has an official segmentation tutorial (LayerGradCam, FeatureAblation) [X6] | IG, aggregated per channel only |
| Perturbation (model-agnostic) | Occlusion / feature ablation over superpixels (SLIC), RISE [X7], MiSuRe (segmentation-specific, sufficient plus counterfactual regions) [X8] | Captum `FeatureAblation` / `Occlusion` with `feature_mask`. RISE is about 40 lines of our own code. | Channel-group occlusion and exact Shapley |
| Mechanistic (model-specific) | Branch ablation, fusion-gate probes, ERF, CKA | own code | Yes |
| Propagation | LRP | zennit 1.0 needs Python ≥3.11.11, so it is **excluded** | No |

#### 2.2.3 Evaluating explanations

The field's consensus is that heatmaps are not evidence until they are evaluated.
**Quantus** [X9] is the standard toolkit. It groups metrics into six families:

| Property | Metrics | Where from |
| --- | --- | --- |
| **Randomisation (sanity)** | MPRT [X10]; its shortcomings [X11]; the fixes **Smooth MPRT and Efficient MPRT** [X12]; Random Logit | Quantus. Our SSIM cascade already exists. |
| **Faithfulness** | Deletion/Insertion AUC [X7]; **ROAD** (noisy-linear imputation, no retraining, avoids the distribution-shift problem) [X13]; Region Perturbation, Pixel Flipping, IROF, Faithfulness Correlation, Infidelity | On this stack: ROAD from **pytorch-grad-cam** (verified on a segmentation target); Region Perturbation and Pixel Flipping from Quantus (verified). Quantus's ROAD, IROF and FaithfulnessCorrelation failed on Python 3.10 + scipy 1.10.1 (§1.4). |
| **Robustness** | Max- and Avg-Sensitivity, Local Lipschitz, RIS/ROS | Quantus |
| **Localisation (plausibility)** | Relevance Mass/Rank Accuracy, Pointing Game, Top-K Intersection, AUC | Quantus |
| **Complexity** | Sparseness, Complexity, Effective Complexity | Quantus |
| **Axiomatic** | Completeness, Non-Sensitivity, Input Invariance | Quantus |

Two caveats specific to segmentation:

- **Localisation measures plausibility, not faithfulness.** A map equal to the predicted mask
  scores near-perfectly against the ground truth whenever the model is accurate. Report the two
  families separately, and never cite localisation as evidence of faithfulness.
- **Quantus's API assumes classification** (it indexes `model(x)` by a label). Wrap the model so
  that it returns the scalar segmentation target as a one-logit output. This wrapper ran
  correctly with 10 Quantus metrics in the smoke test (§1.4). Still, **validate each Quantus
  metric on a toy model with a known answer before trusting it.**

Medical-imaging evidence: several saliency methods fail trustworthiness tests (randomisation,
repeatability) on medical images [X14], and saliency localisation falls short of human
benchmarks on chest X-ray [X15]. That supports the repo's existing rule that saliency is
qualitative unless it passes sanity checks, which Phase 4 extends with faithfulness.

#### 2.2.4 Reporting standards

- **CLAIM 2024** [R1] asks that model details, initialisation and transfer, and interpretability
  methods be reported. Items are answered Yes/No/NA.
- **TRIPOD+AI** [R2] asks for code, model-weight and data availability statements.
- **FUTURE-AI** [R3] lists explainability among the principles for trustworthy medical-imaging AI.

The provenance fields added in Phase 2 (weight source, sha256, licence) and the XAI outputs from
Phase 4 map directly onto those items.

---

# Part B — Plan

## 3 · Principles for all phases

1. **One change type per phase.** Phase 0 moves files, Phase 1 changes versions, Phases 2–5 add
   features. Each phase is verified before and after.
2. **Use a standard library before writing our own code:** timm for encoders and weights, Captum
   and pytorch-grad-cam for attribution, Quantus for evaluation (only the metrics verified on
   this stack). Every library must also pass the Python 3.10 + torch 1.13 smoke test. Our own code stays
   only where it is research-specific (channel groups, Mamba branches, CBFFM) or serves as a
   cross-check.
3. **Provenance is part of identity.** Anything that changes a result (the weights, the fine-tuning
   recipe) goes into `config_hash`. Anything that describes a run (licence, download URL) goes
   into the manifest.
4. **Fail loudly.** No silent random-init fallback, no silent key dropping, no silent
   implementation swap.
5. **The existing guarantees still hold:** test-set tokens, the ledger, multi-seed runs, and
   reporting blocking rules apply to every new feature.

---

## 4 · Phase 0: Repository reorganisation (first, with no behaviour change)

### 4.1 Target layout

Tracked root directories drop from 20 to 7.

```
dissert/
├── .github/workflows/ci.yml
├── configs/                      # unchanged tree (YAML is data, not code; paths stay root-relative)
├── docs/
│   ├── reference.md              # ← CODE_REVIEW.md
│   ├── output-layout.md          # ← OUTPUT_LAYOUT.md
│   └── design/
│       ├── session-grouping.md   # ← SESSION_GROUPING_PLAN.md
│       ├── xdash-resume-contract.md  # ← XDASH_RESUME_CONTRACT.md (commit it first)
│       └── transfer-xai-plan.md  # ← this file
├── notebooks/
├── scripts/
│   └── reproduce.sh              # shell orchestration only
├── src/dissert/
│   ├── __init__.py
│   ├── cli/                      # entry points (thin argparse + main())
│   │   ├── train.py              # ← train.py
│   │   ├── eval.py               # ← eval.py
│   │   ├── search.py             # ← search.py
│   │   └── report.py             # ← scripts/generate_report.py
│   ├── config/
│   │   ├── loader.py             # ← utils/config.py   (compose:-merge)
│   │   └── schema.py             # ← orchestration/schema.py
│   ├── datasets/                 # ← datasets/          (name kept; namespaced, so no HF clash)
│   ├── models/                   # ← models/
│   │   └── params.py             # ← utils/metrics.py   (count_parameters)
│   ├── losses/                   # ← losses/
│   ├── metrics/                  # ← metrics/
│   ├── training/                 # ← training/ + utils/{checkpoint,early_stopping,plot_training}.py
│   ├── evaluation/               # ← utils/report.py (EvaluationReporter, seed aggregation),
│   │                             #   utils/visualize.py (confusion/ROC/PR plots)
│   ├── orchestration/            # ← orchestration/ (minus schema.py)
│   ├── analysis/                 # post-hoc analysis of trained models
│   │   ├── stats/                # ← stats/
│   │   ├── profiling/            # ← profiling/
│   │   ├── uncertainty/          # ← uncertainty/
│   │   ├── robustness/           # ← robustness/
│   │   └── mechanism/            # ← analysis/ (ERF, CKA, failure taxonomy)
│   ├── xai/                      # ← attribution/ (module files unchanged)
│   ├── reporting/                # ← reporting/
│   └── utils/
│       └── logger.py             # ← utils/logger.py (the only true utility left)
├── tests/                        # same files, imports rewritten
├── pyproject.toml                # ← requirements.txt + pytest.ini
├── environment.lock              # ← env/environment.lock
├── README.md
├── CHANGELOG.md
└── .gitignore
# git-ignored runtime roots (not tracked):
#   data/      inputs:  datasets + data/weights/ (manually obtained weights only)
#   outputs/   everything generated (experiments, ledger, reports, cache/)
```

Reasons behind specific choices:

- **`src/` layout with a single namespace.** This is the PyPA-recommended layout. Tests run
  against the installed package, not against whatever happens to be in the working directory,
  and every generic package name is safely namespaced.
- **Package names are kept wherever possible.** `datasets`, `models`, `losses`, `metrics`,
  `training`, `orchestration` and `reporting` keep their names. Only four things are renamed or
  grouped: the `utils/` grab-bag is split; schema and loader form `config/`; the five post-hoc
  packages are grouped under `analysis/`, which matches README §10's existing "Analysis suite"
  grouping; and `attribution/` becomes `xai/` because Phase 4 widens its scope from attribution
  to explanation plus evaluation.
- **`count_parameters` moves to `models/params.py`, not `analysis/profiling`**, so that `models`
  never imports from `analysis`. The dependency direction stays
  `cli → training/evaluation/analysis → models/datasets/metrics → config/utils`.
- **`run_training` stays inside `cli/train.py` in this phase.** Moving it into `training/run.py`
  is a code change, not a file move, so it waits for Phase 1 or later.
- **`configs/` stays at the root** and keeps its layout. External tooling (XDash, notebooks)
  addresses configs by path, and `config_hash` does not depend on file location.

### 4.2 File mapping

| From | To |
| --- | --- |
| `train.py`, `eval.py`, `search.py` | `src/dissert/cli/{train,eval,search}.py` |
| `scripts/generate_report.py` | `src/dissert/cli/report.py` |
| `datasets/`, `models/`, `losses/`, `metrics/`, `training/`, `orchestration/`, `reporting/` | `src/dissert/<same name>/` |
| `stats/`, `profiling/`, `uncertainty/`, `robustness/` | `src/dissert/analysis/<same name>/` |
| `analysis/` | `src/dissert/analysis/mechanism/` |
| `attribution/` | `src/dissert/xai/` |
| `utils/config.py` · `orchestration/schema.py` | `src/dissert/config/{loader,schema}.py` |
| `utils/checkpoint.py`, `utils/early_stopping.py`, `utils/plot_training.py` | `src/dissert/training/` |
| `utils/report.py`, `utils/visualize.py` | `src/dissert/evaluation/{report,plots}.py` |
| `utils/metrics.py` | `src/dissert/models/params.py` |
| `utils/logger.py` | `src/dissert/utils/logger.py` |
| `requirements.txt`, `pytest.ini` | `pyproject.toml` (dependencies copied **verbatim**; version changes wait for Phase 1) |
| `env/environment.lock` | `environment.lock` |
| `CODE_REVIEW.md`, `OUTPUT_LAYOUT.md`, `SESSION_GROUPING_PLAN.md`, `XDASH_RESUME_CONTRACT.md`, this file | `docs/…` as in the tree above |
| root `artifacts/` (untracked, created by the channel-stats cache default) | deleted. The default in `datasets/channels.py:265` changes to `outputs/cache/channel_stats` (the phase's only non-import edit; it affects a cache only). |
| `outputs/{checkpoints,logs,runs,artifacts,kaggle,search_test_results}` (legacy pre-hash layout) | `outputs/_legacy/`, **after the owner reviews it**. Nothing is deleted. |
| `scripts/image.py` (untracked, hard-coded absolute paths) | left to the owner; it is not part of the framework |

### 4.3 `pyproject.toml` essentials

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "dissert"
requires-python = ">=3.8"          # becomes ">=3.10,<3.11" in Phase 1
dependencies = [ ...verbatim from requirements.txt... ]

[project.optional-dependencies]    # filled in Phases 1–4
dev = ["pytest>=8"]

[project.scripts]
dissert-train  = "dissert.cli.train:main"
dissert-eval   = "dissert.cli.eval:main"
dissert-search = "dissert.cli.search:main"
dissert-report = "dissert.cli.report:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`python -m dissert.cli.train --config …` works as well as `dissert-train --config …`. Runs are
still launched from the repo root, because `configs/`, `data/` and `outputs/` stay root-relative
exactly as today.

### 4.4 Steps

1. **Baseline snapshot, before any move:**
   - `pytest -q`: record the pass count. At HEAD `165eb57` it is **357 passed, 2 skipped**; the
     README's "316 tests" is out of date.
   - A small script writes `config_hash` for every `configs/experiment/**/*.yaml` to a scratch
     JSON file.
   - Note one existing `best.pth` to reload afterwards.
2. `git mv` each path in §4.2, so `git log --follow` keeps history.
3. Rewrite imports mechanically with a script. The patterns are
   `^(\s*)(from|import) (datasets|models|…)\b` → `dissert.<new path>`, plus the individual module
   moves (`utils.config` → `dissert.config.loader`, `orchestration.schema` →
   `dissert.config.schema`, and so on). Relative imports inside packages are unaffected. Update
   the `__init__` re-exports in `utils/` and `training/`.
4. Rewrite the **string** references that the import pass misses:
   - `python -m datasets.stats` and `python -m orchestration.sweep` in code, docs and the README;
   - `scripts/reproduce.sh`: drop the `PYTHONPATH` export and call `python -m dissert.cli.*`;
   - the CI workflow: `pip install -e .[dev]`;
   - both notebooks: `pip install -e .`, and the probe cell's `from utils.config import
     load_config` → `from dissert.config.loader import load_config`;
   - `tests/test_orchestration.py`'s `from train import run_training`.
5. Update `.gitignore`: add `*.egg-info/` and `.pytest_cache/`; drop the `image.py` special case
   and `/pretrained_pth`.
6. Docs: fix every cross-link. Remove or restore the dangling `Technical_Framework_Spec.md` and
   `IMPLEMENTATION_PLAN.md` references. Update the README's layout table. Add a CHANGELOG entry.
7. **Coordinate with XDash** (outside this repo). `repos/dissert.yaml`'s `train_script` and
   `eval_script` change from `train.py`/`eval.py` to `src/dissert/cli/train.py`/`…/eval.py`.
   These work as script paths once the package is installed, or XDash can switch to
   `-m dissert.cli.train`. If XDash cannot change in the same window, add two temporary root shims
   (`train.py`: `from dissert.cli.train import main; main()`) and delete them once it has.

### 4.5 Acceptance criteria

- [ ] `pip install -e .[dev] && pytest -q` shows the same pass count as the baseline.
- [ ] Every experiment config's `config_hash` is byte-identical to the snapshot. Existing
  `outputs/experiments/<name>/<hash7>-s<seed>/` directories stay addressable and resumable.
- [ ] A pre-move `best.pth` loads through `dissert-eval`. Checkpoints hold state dicts, not
  pickled classes, so module paths do not matter; the check confirms it.
- [ ] A one-epoch CPU smoke run (`dissert-train --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml --seed 42 --epochs 1`, with `training.device=cpu`) completes.
- [ ] `git grep -nE "^(from|import) (datasets|models|losses|metrics|training|orchestration|reporting|stats|profiling|uncertainty|robustness|analysis|attribution|utils)\b" -- src tests` returns nothing.
- [ ] `ls` at the root shows only the tree in §4.1, plus the git-ignored `data/` and `outputs/`.

---

## 5 · Phase 1: Python 3.10 on torch 1.13.1 (prerequisite for Phases 2–4)

**Target pins.** All of these were verified together (§1.4).

| Package | Pin | Change from today |
| --- | --- | --- |
| Python | 3.10 | 3.8 → 3.10 |
| torch / torchvision | 1.13.1 / 0.14.1 (`+cu117` wheels on GPU, `+cu116` for older drivers, as today) | none |
| numpy, scipy, scikit-learn, scikit-image, pandas | 1.22.4, 1.10.1, 1.3.2, 0.21.0, 2.0.3 | none. numpy must stay <2 because torch 1.13 cannot import under numpy 2. |
| albumentations | 1.1.0 | none |
| opencv-python **and** opencv-python-headless | both 4.11.0.86 | **forced downgrade from 5.0.0.93.** opencv 5.0.0.93 declares `numpy>=2` on Python ≥3.9. Pin both packages to the same version, because both install the `cv2` module and grad-cam depends on the headless one. |
| timm | 1.0.30 | 0.6.12 → 1.0.30 |
| huggingface-hub | current (2.0.0 at the time of the dry run) | 0.11.0 → current |
| captum | 0.8.0 | today unpinned, and it resolves to 0.7.0 on Python 3.8 |
| mamba-ssm / causal-conv1d | 1.0.1 / 1.1.1, `cu118torch1.13…cp310` GitHub-release wheels | the wheel tag only |

**Why Phase 1 comes before the features.** captum 0.8, segmentation-models-pytorch 0.5 and
huggingface-hub 2.x need Python ≥3.9/3.10. timm ≥1.0 is needed for the DINOv3, Hiera, SAM-ViT
and ConvNeXt-V2 weights. **Kaggle:** in both notebooks change the deadsnakes `python3.8` venv to
`python3.10`. A venv is still needed, because Kaggle's stock image ships torch 2.x.

**Dry-run results (2026-09-25).** HEAD `165eb57` was copied to a scratch directory and run on
CPU with no code changes:

| Check | Python 3.8 / timm 0.6.12 / opencv 5.0 (today) | Python 3.10 / timm 1.0.30 / opencv 4.11 (target) |
| --- | --- | --- |
| Test suite | 357 passed, 2 skipped | **357 passed, 2 skipped** |
| Model initialisation: sha256 of the `state_dict` for all 13 `configs/experiment/**` models, seed 0 | reference | **identical for all 13** |
| Forward output of those 13 models on a fixed input | reference | **identical for all 13** |
| cv2 PNG decode; resize (linear, nearest, area); RGB→YCrCb and RGB→gray | reference | identical |
| albumentations `HorizontalFlip` and `RandomBrightnessContrast` (same seed) | reference | identical |
| albumentations `ShiftScaleRotate`, which uses `cv2.warpAffine` | reference | **different: 62% of pixels, max \|Δ\| = 6/255** |

Known break points:

| Break | Where | Fix |
| --- | --- | --- |
| `cv2.warpAffine` output changed between opencv 5.0 and 4.11 | Training augmentation (`ShiftScaleRotate`). The robustness geometric perturbations use torch `grid_sample` and are unaffected. The JPEG and blur corruptions have not been compared yet. | **Training on 3.10 is not bit-identical to training on 3.8.** Produce every result that goes into the dissertation on the new stack. Add a reporting blocking rule that refuses to put runs whose manifests record different Python minor, torch or opencv versions into one table. Existing 3.8 runs stay valid as their own set. |
| Deprecated timm import paths: `timm.models.layers`, `timm.models.registry`, `timm.models.helpers` | `models/pvtv2.py`, `models/baseline/*` | Only a FutureWarning today. Switch to `timm.layers` / `timm.models`. |
| Python 3.10 reaches end of life in October 2026 | platform | Acceptable within a dissertation timeline. Exact pins plus `environment.lock` keep it reproducible; security fixes stop. |

The torch 2.x break points from rev. 1 (the `weights_only` default and the AMP API deprecation)
no longer apply, because torch stays at 1.13.1, where `torch.load` still defaults to
`weights_only=False`. External weights should still load through safetensors, never through a
pickle.

**Dependency cleanup:** remove the 15 never-imported packages listed in §1.1. `pyarrow` and `onnx`
stay, because pandas and torch.onnx use them indirectly. Regenerate `environment.lock` with
`pip freeze` on the new stack. Set the CI `python-version` to `3.10`.

**Gate:**
- the suite passes in CI on 3.10;
- one real GPU training run per model family finishes on Kaggle;
- the Mamba fused kernel (cp310 wheel) and the reference scan agree on GPU (the existing
  `test_mamba.py` check).

---

## 6 · Phase 2: Pretrained-weight infrastructure

### 6.1 One encoder factory (replaces `models/backbones.py`)

```python
# src/dissert/models/encoders.py
def build_encoder(spec: EncoderSpec, in_chans: int) -> Encoder:
    """timm.create_model(spec.name, features_only=True, out_indices=spec.out_indices,
    pretrained=False), then load_weights(spec.weights), then adapt the input stem to in_chans.
    Returns the module, per-stage channels (deepest first, as EMCAD expects today),
    per-stage strides, and the pretrained_cfg (mean/std)."""
```

- **Hierarchical encoders** (PVTv2, ResNet, ConvNeXt(-V2), EfficientNet, Hiera, MambaOut) plug
  into the existing decoders directly. On torch 1.13 each returned 4–5 maps at strides 4–32 (2–32
  for ResNet and EfficientNet) and backpropagated (§1.4).
- **Plain ViTs** (DINOv3, SAM-ViT) go through timm ≥1.0's `features_only`/`forward_intermediates`
  plus a **simple feature pyramid neck** (deconvolution/pooling to strides 4/8/16/32, the
  ViTDet-style neck), so they fit the same decoders. The neck exists only because single-scale
  ViTs need it [T11]. On torch 1.13, `features_only` returned three stride-16 maps (384 channels
  for DINOv3-S, 768 for SAM-B), which confirms the neck is needed. Without fused attention on
  torch 1.13, memory use at 352² should be measured before committing to ViT-B sizes.
- **Delete the vendored `models/pvtv2.py` and `models/resnet.py`**, since timm provides both,
  once the parity check in §6.5 passes. If exact reproduction of the original EMCAD weights is
  needed, keep a `file:` weight source (below) rather than keeping the vendored code.
- **No broad `except`, and no fallback to a different implementation.**

### 6.2 Weight specification and provenance

Config (strict, schema-validated; replaces EMCAD's `pretrain`/`pretrained_dir`):

```yaml
model:
  name: emcad
  encoder:
    name: pvt_v2_b2
    weights: timm:pvt_v2_b2.in1k        # timm:<tag> | hf:<repo>@<revision> | file:<path> | none
    sha256: null                         # REQUIRED for file: sources (validated at load)
    in_chans_strategy: zero_init         # zero_init | separate_stem | repeat_scale
    normalization: pretrained            # pretrained | dataset
```

- The `weights` string, the `sha256` and the strategy are ordinary config fields, so **they are
  part of `config_hash` automatically**. A run with other weights gets another directory. `hf:`
  sources must pin a revision (a commit hash) so the identifier cannot silently change content.
- `load_weights` returns a `WeightRecord`: source, resolved revision, sha256 of the loaded
  tensors, licence (from the Hub card), and missing and unexpected keys. The record is written
  into the run manifest and into checkpoint metadata.
- **Strict by default.** Any missing or unexpected key raises. An explicit
  `allow_missing: ["head.", …]` list covers heads that are replaced on purpose. A missing file
  raises instead of falling back to random weights.
- Weights are cached in the Hugging Face cache (`HF_HOME`; on Kaggle, point it at an attached
  dataset to avoid re-downloading). Manually obtained weights live in `data/weights/`, which is
  git-ignored.

### 6.3 Input-stem adaptation (§2.1.3)

- Use timm's `pretrained_cfg["first_conv"]` to find the stem convolution of any timm encoder.
- `zero_init` (default): new conv with `in_chans`; `W[:, rgb] = W_pretrained`; `W[:, extra] = 0`.
  Needs the channel-group slice map, which `xai/common.resolve_group_slices` already computes, to
  know where RGB sits.
- `separate_stem`: `stem_pretrained(x_rgb) + stem_scratch(x_extra)`, with `stem_scratch`'s
  output zero-initialised as well.
- `repeat_scale`: delegates to timm's `adapt_input_conv`. Ablation arm only.
- Grayscale modalities (BUSI ultrasound) use timm's built-in `in_chans=1` handling, which sums
  the RGB kernels. That is standard and correct for single-channel input.

### 6.4 Wire-up

- EMCAD switches to `build_encoder`.
- Add **one new registered family, `encoder_unet`**: any `build_encoder` output plus the existing
  UNet decoder (`models/decoder.py`/`blocks.py`). It gives a fixed-decoder test bed for comparing
  pretraining sources, the controlled comparison §2.1.1 calls for. No other new architectures.
- Optionally, register `smp_unet`, a thin wrapper around `segmentation_models_pytorch.Unet` with
  `encoder_name="tu-<timm name>"`, as a library reference baseline that reviewers will recognise.
  Add it only if a reviewer-facing baseline is actually wanted. smp 0.5.0 was verified on
  Python 3.10 + torch 1.13, including `in_channels=5`.
- `ModelRegistry.get()` reports **both trainable and total** parameters, and `budget_ceiling`
  applies to total unless `budget_on: trainable` is set explicitly.
- Normalisation: `datasets` takes RGB mean/std from `encoder.pretrained_cfg` when
  `normalization: pretrained`, and the statistics actually used go into the manifest.

### 6.5 Tests

- Parity: the old PVT loader and `build_encoder("pvt_v2_b2", "timm:pvt_v2_b2.in1k")` produce
  matching stage features on a fixed input, allowing for key remapping. This check decides
  whether timm's port can replace the vendored code.
- A missing weights file raises. A spurious key raises. `allow_missing` works.
- `config_hash` changes when `weights` changes.
- `zero_init`: the model output on `[rgb, extra]` equals the pretrained model's output on `rgb`
  at initialisation, whatever values `extra` holds.
- `normalization: pretrained` picks the `pretrained_cfg` statistics.

---

## 7 · Phase 3: Transfer-learning strategies

Keep it small. Most strategies are **recipes built on the existing `stages` mechanism**, not new
code paths.

| Need | Implementation |
| --- | --- |
| Discriminative LR | `training.lr_mult: {"backbone": 0.1}`, where a module-name prefix maps to an LR multiplier. It is applied in `build_optimizer` on top of the existing decay/no-decay split (`test_param_groups` extends to cover it). |
| Layer-wise LR decay (ViTs) | `training.layer_decay: 0.75`. Block depth comes from timm's `group_matcher`, which exists for exactly this purpose. |
| LP-FT, gradual unfreezing, surgical fine-tuning | Recipes on `stages`, extended with a per-stage `lr_mult`. Example: stage 1 `freeze: ["backbone"]`, stage 2 unfrozen with `lr_mult: {backbone: 0.1}`. Ship them as `configs/transfer/*.yaml` fragments. No `strategy:` enum. |
| BN statistics | `stages[].freeze_bn: true` puts BN layers inside frozen modules into `eval()`. Uses timm's `freeze_batch_norm_2d` or a small hook. |
| Cross-dataset fine-tuning | `model.init_from: <run_id or checkpoint path>`: strict load (EMA weights preferred, as eval does now), with heads re-initialised when `out_channels` differs. The source `run_id` goes into the manifest, and the source checkpoint's sha256 goes into `config_hash`. |
| L2-SP | One loss term (`l2sp`) that holds a frozen copy of the initial encoder weights. **Optional; add only if the fine-tuning ablation shows drift.** |
| LoRA / PEFT | Only when ViT foundation encoders (DINOv3, SAM-ViT) are in the study. On torch 1.13, current `peft` is not usable, because its accelerate (≥1.0) and transformers (≥4.49) dependencies target torch ≥2.0. Pinning an old peft/transformers/accelerate trio is untested and fragile. **Write a `LoRALinear` wrapper** instead (frozen `W` plus trainable `B·A`, scaled by `alpha/r`, with `B` zero-initialised), about 30 lines, and apply it to the ViT's `qkv` layers. Test: output unchanged at initialisation, and only A and B receive gradients. |

Tests: the parameter-group LR multipliers match the config; frozen parameters get no gradient;
BN running statistics do not change in a frozen stage; `init_from` rejects a shape mismatch
outside the allowed heads; resuming mid-stage restores the frozen state. The existing resume logic
restores stage state, and the test must confirm that `freeze_bn` survives resume.

---

## 8 · Phase 4: Explainable AI

### 8.1 Package shape (`src/dissert/xai/`)

```
xai/
├── targets.py        # NEW  SegTarget: which pixels (pred | gt | roi | boundary | all), on logits|probs, sum
├── common.py         # existing (group slices, training-mean baseline, occlusion primitive)
├── occlusion.py  shapley.py  integrated_grads.py  branch.py  fusion_probe.py   # existing
├── segcam.py         # existing own Seg-Grad-CAM / Seg-XRes-CAM (kept as reference implementations)
├── cams.py           # NEW  pytorch-grad-cam adapter: GradCAM, HiResCAM, GradCAM++, LayerCAM,
│                     #      ScoreCAM, AblationCAM, EigenCAM, SegEigenCAM (all verified with grad-cam 1.5.7)
├── gradients.py      # NEW  Captum 0.8.0 adapter: pixel-level IG, NoiseTunnel(SmoothGrad), GradientSHAP
├── perturbation.py   # NEW  Captum FeatureAblation over SLIC superpixels (skimage), RISE
├── evaluate.py       # NEW  faithfulness / randomisation / localisation / complexity metrics
└── sanity.py         # existing MPRT (SSIM cascade) + label randomisation; + sMPRT/eMPRT via Quantus
```

- **`SegTarget` is the one abstraction.** It turns `model(x)` into a scalar per image. Captum
  gets it as the `forward_func` reduction; pytorch-grad-cam gets it as a target callable matching
  `SemanticSegmentationTarget`'s contract; Quantus gets it through the one-logit wrapper (§2.2.3).
  The existing `integrated_grads._foreground_mass_forward` becomes `SegTarget(pixels="all",
  on="probs")`, so its behaviour is unchanged.
- **Target layers are declared per model family** in the registry, for example
  `xai_layers = {"decoder_last": "...", "bottleneck": "..."}`, so CAMs never guess. ViT encoders
  also declare their `reshape_transform`.
- **Cross-checks against our own implementation, as tests:** our `seg_grad_cam` must match
  pytorch-grad-cam `GradCAM` with the same target and layer, and our `seg_xres_cam` must match
  `HiResCAM`. Both follow from how the methods are defined, so a mismatch means a bug on one side.
  Seg-HiRes-Grad CAM [X3] is added only if it turns out to differ from `seg_xres_cam` at the
  implementation level; check the paper's code first.
- **Mamba models:** gradients through the fused selective-scan kernel are not guaranteed to match
  the reference scan's. Run gradient methods with `scan_impl=reference` (already selectable) and
  record it. Gradient-free CAMs and perturbation methods avoid the issue.

### 8.2 Evaluation (`xai/evaluate.py`)

| Metric | Implementation | Target used |
| --- | --- | --- |
| Deletion / Insertion AUC (MoRF / LeRF) | own (about 60 lines, batched) | SegTarget score **and** Dice of the explained mask |
| ROAD (MoRF/LeRF, noisy linear imputation) | pytorch-grad-cam `ROADMostRelevantFirst/LeastRelevantFirst/Combined` (verified with `SemanticSegmentationTarget`). Quantus's `ROAD` failed on this stack. | SegTarget score |
| Region Perturbation, Pixel Flipping | Quantus through the wrapper (verified) | SegTarget score |
| Faithfulness Correlation | **Our own** (per-image Pearson between attribution sums and score drops over random subsets, about 15 lines). Quantus's version calls `pearsonr(axis=…)`, which needs scipy ≥1.14 and therefore numpy ≥1.23.5, a numeric-stack change not worth making for one metric. | SegTarget score |
| IROF | Dropped. Quantus's IROF raised a shape error in the smoke test; add it back only if that is resolved. | n/a |
| MPRT (cascade), **sMPRT, eMPRT** | our SSIM version, plus Quantus `MPRT`/`SmoothMPRT`/`EfficientMPRT` (all verified) | n/a |
| Max-Sensitivity | Quantus | n/a |
| Relevance Mass Accuracy, Pointing Game (**plausibility**) | Quantus | ground truth and prediction, reported separately |
| **Context-reliance ratio**: attribution mass outside the dilated GT mask | own (5 lines) | Links to the existing shortcut audit (`robustness.geometric`) |
| Sparseness / Complexity | Quantus | n/a |
| Method agreement (rank correlation) | existing `agreement_score`, generalised to maps | n/a |

**Quantus rule:** each Quantus metric is enabled only after a test on a toy model with a known
answer passes (§2.2.3).

### 8.3 CLI and outputs

```
dissert-explain --config <exp.yaml> [--seed 42] --split val \
    --methods gradcam hirescam ig smoothgrad rise --target pred --n-images 50 \
    [--metrics deletion insertion road mprt]
```

- Writes to `outputs/experiments/<exp>/<hash7>-s<seed>/xai/<split>/`:
  - `maps/<method>.npz` (float16);
  - `metrics.parquet` (one row per image × method × metric);
  - `sanity.json` (per-method pass or fail, with thresholds);
  - `panels/*.png`;
  - `xai_manifest.json` (methods, layers, target, `scan_impl`, library versions).
  `docs/output-layout.md` gets updated to match.
- **The default split is validation.** `--split test` needs a test token and is logged in the
  ledger, like every other test touch.
- **Reporting rule, extended:** a saliency figure is refused unless its method has a passing
  `sanity.json` **and** faithfulness metrics exist in `metrics.parquet`. This builds on the
  existing "saliency sanitised" rule.
- `scripts/reproduce.sh`'s XAI stage stops printing function names and calls `dissert-explain`.

### 8.4 Dependencies

Add an optional extra `[xai]` with `captum==0.8.0` (already a dependency; today it is unpinned),
`grad-cam==1.5.7` and `quantus==0.6.0`. These are exactly the versions verified on Python 3.10 +
torch 1.13.1 with numpy 1.22.4 and scipy 1.10.1 (§1.4). Pin them in `environment.lock` as well.
Don't let pip float them: a newer captum requires torch ≥2.3.

---

## 9 · Phase 5: Study protocol and reporting

This part is about experiment design; it adds only configs and report code.

**Transfer study** (a fixed decoder isolates the effect of initialisation):

| Factor | Levels |
| --- | --- |
| Model | `encoder_unet` with a fixed decoder; EMCAD; MK-UNet/GMK-UNet via `init_from` only (no public weights exist for them) |
| Initialisation | scratch · ImageNet supervised · ImageNet SSL (ConvNeXt-V2 FCMAE or Hiera MAE) · DINOv3 (ViT-S/B with neck) · domain-specific if weights are obtainable (RadImageNet for BUSI; EndoDINO or GastroNet for polyps) · cross-dataset `init_from` |
| Strategy | frozen encoder · full fine-tuning · LP-FT |
| Input stem (m2–m5 only) | `zero_init` · `separate_stem` · `repeat_scale` |
| Evaluation | in-distribution test **and** cross-dataset without fine-tuning (ClinicDB → ColonDB), because LP-FT's claimed benefit is out of distribution [T16] |

Seeds, test-token discipline and significance testing reuse the existing machinery: the 3-seed
default, `stats.run_family_comparison`, and Holm correction within a declared family. Report
**trainable and total** parameters and FLOPs.

**XAI study:**

1. Do the methods pass sanity checks and measure as faithful on these models? Compare MPRT with
   sMPRT/eMPRT, and deletion/ROAD across methods.
2. Does pretraining change what the model relies on? Compare the channel-group Shapley shift and
   the context-reliance ratio between scratch and pretrained models of the same architecture.
   Question 2 connects Phases 2–4 into one research question.

**New report artefacts:**

- the transfer matrix table (initialisation × strategy, mean ± std, significance markers);
- the ID-vs-OOD scatter;
- the XAI faithfulness table (method × metric);
- sanity-check panels.

All of them read only from artefacts, as the reporting contract requires.

**New blocking rules:**

- mixed environments in one table: different Python minor, torch or opencv versions (from Phase 1);
- a pretrained run whose `WeightRecord` is missing;
- saliency without faithfulness metrics (from Phase 4).

### Deferred (YAGNI until evidence says otherwise)

| Item | Why deferred | Add when |
| --- | --- | --- |
| In-domain SSL pretraining: SparK for CNNs (masked modelling that works on any CNN/U-Net encoder, ICLR 2023 [T22]), MAE, DINO | Costly on Kaggle's session budget, and the evidence shows in-domain SSL does not reliably beat ImageNet SSL on polyp generalisation [T6] | The transfer study shows no pretrained source helps MK-UNet/GMK-UNet, and unlabelled in-domain data is available (for example HyperKvasir-unlabelled) |
| LoRA on ViT encoders (hand-written, §7) | Only meaningful once DINOv3 or SAM-ViT is in the study | The frozen-encoder and full fine-tuning arms leave a gap worth closing |
| VMamba ImageNet weights for the Mamba auxiliary branch | The repo's VSS stage dimensions are custom, so key mapping may be impossible | A 1-day feasibility spike: compare state-dict shapes first |
| Concept-based (TCAV), counterfactual generators, attention rollout | Weak fit for binary lesion segmentation. Attention rollout is not a faithful explanation. | Specific research need |

---

# Part C — Risks, decisions, references

## 10 · Risk register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Test-set contamination through pretraining data.** Endoscopy and medical foundation models (EndoDINO, GastroNet, RadImageNet, BiomedCLIP) may have been pretrained on public sets that include ClinicDB, ColonDB, ISIC or BUSI images. | Inflated test scores, invalid claims | Before adopting any domain-specific weight source, read its pretraining dataset list and record it in the `WeightRecord`. Exclude sources that overlap a test set, or report the overlap explicitly. |
| The Phase 1 move changes training augmentation (opencv 5.0 → 4.11 changes `warpAffine`) | Runs from 3.8 and 3.10 are not bit-comparable. Models, forward passes, resizing and colour conversion are identical (verified). | Produce the dissertation's results on the 3.10 stack; the blocking rule stops mixing environments. |
| **Frozen on torch 1.13** while the ecosystem moves to torch ≥2 | A future library cannot be installed, or a floating dependency breaks the environment | Pin every version (§5, §8.4) and commit `environment.lock`. Any new dependency must first pass a Python 3.10 + torch 1.13 smoke test. Add one in Phase 1: a short script that builds the planned encoders and runs the XAI methods, as was done for §1.4. Never let numpy reach 2.x. |
| No fused attention on torch 1.13 | ViT encoders are slower and use more memory | Prefer hierarchical encoders; measure ViT memory at the target resolution before planning runs. |
| Python 3.10 end of life (October 2026) | No security fixes | Acceptable for an offline research codebase; revisit after the dissertation. |
| Licences (DINOv3 custom licence) | Restrictions on publishing derived weights | Record the licence in the `WeightRecord`. Check before releasing weights (TRIPOD+AI asks about weight availability). |
| GPU memory on Kaggle (16 GB) with ViT-B/L encoders at 352² input | OOM, longer sessions | Prefer S/B sizes; use gradient accumulation (exists) and the frozen-encoder strategy; `--max-hours` budgets already exist. |
| XDash coupling to script paths and `utils.config` | Automated runs break after Phase 0 | §4.4 step 7: coordinated change or temporary shims. |
| Quantus metrics assume classification | Misleading XAI numbers | Toy-model validation gate (§8.2). |
| Gradients through the fused Mamba kernel | Unreliable gradient maps | `scan_impl=reference` for gradient methods (§8.1). |

## 11 · Decisions needed from the owner before starting

1. **XDash:** update `repos/dissert.yaml` and the worker template in the same window as Phase 0,
   or keep temporary root shims?
2. **Legacy outputs:** confirm that `outputs/{checkpoints,logs,runs,artifacts,kaggle,search_test_results}`
   can move to `outputs/_legacy/`.
3. **Python 3.10 results:** agree that every result that goes into the dissertation is produced
   on the Python 3.10 stack. Models are bit-identical, but opencv 4.11 changes `ShiftScaleRotate`
   augmentation slightly, so 3.8 training runs are not bit-comparable.
4. **Foundation-model arm:** DINOv3 and SAM-ViT (both verified on torch 1.13) in scope or not,
   given licence and compute. SAM2 and MedSAM2 are excluded by the platform. Also, can any
   domain-specific weights (EndoDINO, GastroNet, RadImageNet) actually be obtained?
5. **`XDASH_RESUME_CONTRACT.md`** is untracked. Commit it into `docs/design/`, or keep it
   outside the repo?

## 12 · References

**Transfer learning and pretrained weights**

- [T1] Raghu, Zhang, Kleinberg, Bengio. *Transfusion: Understanding Transfer Learning for Medical Imaging.* NeurIPS 2019. https://arxiv.org/abs/1902.07208
- [T2] He, Girshick, Dollár. *Rethinking ImageNet Pre-training.* ICCV 2019. https://arxiv.org/abs/1811.08883
- [T3] Hosseinzadeh Taher et al. *A Systematic Benchmarking Analysis of Transfer Learning for Medical Image Analysis.* DART @ MICCAI 2021. https://arxiv.org/abs/2108.05930
- [T4] Mei et al. *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning.* Radiology: AI 2022. https://pubs.rsna.org/doi/full/10.1148/ryai.210315
- [T5] *Transfer learning with RadImageNet in medical imaging AI: review and future directions.* Neural Computing and Applications 2026. https://link.springer.com/article/10.1007/s00521-026-12435-y
- [T6] Sanderson, Matuszewski. *Polyp Segmentation Generalisability of Pretrained Backbones.* 2024. https://arxiv.org/abs/2405.15524
- [T7] Dermyer et al. *EndoDINO: A Foundation Model for GI Endoscopy.* 2025. https://arxiv.org/abs/2501.05488
- [T8] *GastroNet-5M: A Multicenter Dataset for Developing Foundation Models in GI Endoscopy.* Gastroenterology 2025. https://www.gastrojournal.org/article/S0016-5085(25)05797-X/fulltext
- [T9] *DINO-MVR: Multi-View Readout of Frozen DINOv3 for Annotation-Efficient Medical Segmentation.* 2026. https://arxiv.org/abs/2605.07221
- [T10] *Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation.* 2025. https://arxiv.org/abs/2508.20909
- [T11] Li et al. *MedDINOv3: How to adapt vision foundation models for medical image segmentation?* 2025. https://arxiv.org/abs/2509.02379 · DINOv3: Siméoni et al. 2025. https://arxiv.org/abs/2508.10104
- [T12] Wang et al. *Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train (Endo-FM).* MICCAI 2023. https://arxiv.org/abs/2306.16741
- [T13] *Fine-Tuning SAM2 for Generalizable Polyp Segmentation with a Channel Attention-Enhanced Decoder.* 2025. https://www.researchgate.net/publication/393227725 · *An efficient fine tuning strategy of SAM for polyp segmentation.* Sci. Rep. 2025. https://www.nature.com/articles/s41598-025-97802-w
- [T14] Wu et al. *Medical SAM Adapter.* 2023. https://arxiv.org/abs/2304.12620 · *Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions.* 2025. https://arxiv.org/abs/2511.01284
- [T15] Fan et al. *PraNet: Parallel Reverse Attention Network for Polyp Segmentation.* MICCAI 2020. https://arxiv.org/abs/2006.11392
- [T16] Kumar et al. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022. https://arxiv.org/abs/2202.10054
- [T17] Howard, Ruder. *Universal Language Model Fine-tuning for Text Classification (ULMFiT).* ACL 2018. https://arxiv.org/abs/1801.06146
- [T18] Lee et al. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR 2023. https://arxiv.org/abs/2210.11466
- [T19] Li, Grandvalet, Davoine. *Explicit Inductive Bias for Transfer Learning with Convolutional Networks (L2-SP).* ICML 2018. https://arxiv.org/abs/1802.01483
- [T20] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
- [T21] timm `adapt_input_conv` behaviour and its RGB-only limitation. https://github.com/huggingface/pytorch-image-models/issues/2445
- [T22] Tian et al. *Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling (SparK).* ICLR 2023. https://arxiv.org/abs/2301.03580

**Explainability**

- [X1] Vinogradova, Dibrov, Myers. *Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping (Seg-Grad-CAM).* AAAI 2020. https://arxiv.org/abs/2002.11434
- [X2] Hasany, Petitjean, Mériaudeau. *Seg-XRes-CAM: Explaining Spatially Local Regions in Image Segmentation.* CVPRW 2023. https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/papers/Hasany_Seg-XRes-CAM_Explaining_Spatially_Local_Regions_in_Image_Segmentation_CVPRW_2023_paper.pdf
- [X3] Rheude, Wirtz, Kuijper, Wesarg. *Leveraging CAM Algorithms for Explaining Medical Semantic Segmentation (Seg-HiRes-Grad CAM).* MELBA 2024. https://arxiv.org/abs/2409.20287
- [X4] Gildenblat et al. *pytorch-grad-cam* (methods, `SemanticSegmentationTarget`, ROAD metrics, ViT `reshape_transform`). https://github.com/jacobgil/pytorch-grad-cam · segmentation tutorial: https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.html
- [X5] Kokhlikyan et al. *Captum: A unified and generic model interpretability library for PyTorch.* 2020. https://arxiv.org/abs/2009.07896 · algorithms: https://captum.ai/docs/attribution_algorithms
- [X6] Captum tutorial: *Interpreting semantic segmentation models.* https://captum.ai/tutorials/Segmentation_Interpret
- [X7] Petsiuk, Das, Saenko. *RISE: Randomized Input Sampling for Explanation of Black-box Models* (deletion/insertion metrics). BMVC 2018. https://arxiv.org/abs/1806.07421
- [X8] *MiSuRe is all you need to explain your image segmentation.* 2024. https://arxiv.org/abs/2406.12173
- [X9] Hedström et al. *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations.* JMLR 2023. https://github.com/understandable-machine-intelligence-lab/Quantus
- [X10] Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018. https://arxiv.org/abs/1810.03292
- [X11] Binder et al. *Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations.* CVPR 2023. https://arxiv.org/abs/2211.12486
- [X12] Hedström et al. *A Fresh Look at Sanity Checks for Saliency Maps* (sMPRT, eMPRT). xAI 2024. https://arxiv.org/abs/2405.02383
- [X13] Rong et al. *A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD).* ICML 2022. https://arxiv.org/abs/2202.00449
- [X14] Arun et al. *Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging.* Radiology: AI 2021. https://pubs.rsna.org/doi/10.1148/ryai.2021200267
- [X15] Saporta et al. *Benchmarking saliency methods for chest X-ray interpretation.* Nature Machine Intelligence 2022. https://www.nature.com/articles/s42256-022-00536-x
- Reviews: *XAI in medical imaging: a systematic review of techniques, applications, and challenges.* BMC Medical Imaging 2025. https://link.springer.com/article/10.1186/s12880-025-02118-w · *Which XAI methods in medical imaging are clinically impactful?* Frontiers in AI 2026. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1819422/full

**Reporting standards**

- [R1] Tejani et al. *Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update.* Radiology: AI 2024. https://pubs.rsna.org/doi/full/10.1148/ryai.240300
- [R2] Collins et al. *TRIPOD+AI statement.* BMJ 2024. https://www.bmj.com/content/385/bmj-2023-078378
- [R3] Lekadir et al. *FUTURE-AI: international consensus guideline for trustworthy and deployable AI in healthcare.* BMJ 2025. https://arxiv.org/abs/2309.12325

**Verified stack (installed and exercised, 2026-09-25):** Python 3.10.21 · torch 1.13.1 ·
torchvision 0.14.1 · numpy 1.22.4 · scipy 1.10.1 · scikit-learn 1.3.2 · scikit-image 0.21.0 ·
pandas 2.0.3 · albumentations 1.1.0 · opencv-python(-headless) 4.11.0.86 · timm 1.0.30 ·
huggingface-hub 2.0.0 · captum 0.8.0 · grad-cam 1.5.7 · quantus 0.6.0 ·
segmentation-models-pytorch 0.5.0. **Checked on PyPI or GitHub only:** mamba-ssm 1.0.1 and
causal-conv1d 1.1.1 `cu118torch1.13…cp310` release wheels exist. Floors that rule packages out:
SAM2 needs torch ≥2.5.1, captum 0.9 torch ≥2.3, accelerate ≥1.0 (a peft dependency) torch ≥2.0,
transformers ≥4.49 declares torch ≥2.0, zennit 1.0 Python ≥3.11.11, MONAI 1.6 torch ≥2.8, opencv 5.0 numpy ≥2 on
Python ≥3.9. timm's CI lower bound is "PyTorch 1.13 + Python 3.10" (timm README).
**HF Hub weights checked:** `timm/pvt_v2_b*.in1k` (b0 downloaded and loaded on torch 1.13),
`timm/convnextv2_*.fcmae` (atto downloaded and loaded), `timm/vit_{small,base}_patch16_dinov3.lvd1689m`,
`timm/samvit_*_patch16.sa1b`, `timm/hiera_*_224.mae`, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
Architectures were built and backpropagated on torch 1.13; for DINOv3, SAM-ViT and Hiera the
pretrained weights themselves were not downloaded in the dry run.
