"""
tests/test_encoders.py — Phase 2 (docs/design/transfer-xai-plan.md §6.5):
models/encoders.py's build_encoder factory.

Which tests exercise real downloaded weights vs. local fakes
--------------------------------------------------------------
- test_pvt_v2_b0_parity_vendored_vs_timm_port: no network. Recovers the
  vendored PVTv2 source from git history (commit 3385914, the last commit
  before it was deleted in this phase) and runs the comparison in an
  isolated subprocess — see tests/_pvt_parity_probe.py's docstring for why
  a subprocess is required (importing the vendored module clobbers timm's
  global model registry for pvt_v2_b0..b5).
- test_build_encoder_real_pretrained_download_pvt_v2_b2: real network. Marked
  slow (~40s on first run, cached under ~/.cache/huggingface afterwards).
- Every other test uses either weights: none (random init, no I/O) or a
  small locally-constructed file: checkpoint (harvested from an unpretrained
  resnet18's own state dict — see _make_fake_checkpoint) — no network.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest
import torch

from dissert.config.schema import validate_config
from dissert.models.encoders import build_encoder, resolve_pretrained_norm_stats
from dissert.orchestration.runid import config_hash

# The last commit before models/pvtv2.py was deleted in this phase — see
# module docstring. A fixed, permanent reference into git history, not the
# moving HEAD (which no longer has the file after this phase's commit).
_PRE_DELETION_REV = "3385914"
_PROBE_SCRIPT = Path(__file__).with_name("_pvt_parity_probe.py")


# ---------------------------------------------------------------------------
# Parity: vendored PVTv2 vs. timm's port (decides whether pvtv2.py could be
# deleted — it was, after this test was run and found passing).
# ---------------------------------------------------------------------------

def test_pvt_v2_b0_parity_vendored_vs_timm_port(tmp_path):
    vendored_src = subprocess.run(
        ["git", "show", f"{_PRE_DELETION_REV}:src/dissert/models/pvtv2.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True,
    ).stdout
    vendored_path = tmp_path / "_vendored_pvtv2.py"
    vendored_path.write_text(vendored_src)

    result = subprocess.run(
        [sys.executable, str(_PROBE_SCRIPT), str(vendored_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["shapes_old"] == payload["shapes_new"]
    # Allowing for key remapping (patch_embed1.* -> patch_embed.*,
    # blockN.* -> stages_{N-1}.blocks.*, mlp.dwconv.dwconv.* ->
    # mlp.dwconv.*, normN.* -> stages_{N-1}.norm.*), transplanting the
    # vendored model's own weights into timm's port and running both on the
    # same fixed input gives bit-identical stage features (float rounding
    # only, ~1e-6) — confirms timm's port is a faithful reimplementation.
    assert all(d < 1e-4 for d in payload["diffs"]), payload


def test_build_encoder_pvt_channels_match_old_hardcoded_decoder_contract():
    """Structural regression standing in for the vendored code once it's
    gone: models/backbones.py used to hardcode channels=[512,320,128,64]
    for every pvt_v2_b1..b5 variant and [256,160,64,32] for b0 — the exact
    per-stage widths EMCAD's decoder is built around (models/baseline/
    emcad.py). build_encoder must still produce that same contract."""
    b2 = build_encoder({"name": "pvt_v2_b2", "weights": "none"}, in_chans=3)
    assert b2.channels == [512, 320, 128, 64]
    assert b2.strides == [32, 16, 8, 4]

    b0 = build_encoder({"name": "pvt_v2_b0", "weights": "none"}, in_chans=3)
    assert b0.channels == [256, 160, 64, 32]


@pytest.mark.slow
def test_build_encoder_real_pretrained_download_pvt_v2_b2():
    """Real network + real timm/pvt_v2_b2.in1k download (cached under
    ~/.cache/huggingface after the first run). Confirms build_encoder's
    strict-loading path works end to end against an actual published
    checkpoint, not just a local fake."""
    encoder = build_encoder({"name": "pvt_v2_b2", "weights": "timm:pvt_v2_b2.in1k"}, in_chans=3)
    rec = encoder.weight_record
    assert rec.source == "timm:pvt_v2_b2.in1k"
    assert rec.sha256 and len(rec.sha256) == 64
    assert rec.licence == "apache-2.0"
    assert rec.resolved_revision == "timm/pvt_v2_b2.in1k"
    # Only the classification head is absent — see encoders.py's
    # _build_pretrained_timm docstring on why that's always true for a
    # features_only + classification-pretrained combination, not a
    # per-config exception.
    assert rec.missing_keys == ["head.weight", "head.bias"]
    assert rec.unexpected_keys == []

    x = torch.randn(1, 3, 64, 64)
    outs = encoder.module(x)  # module() itself stays shallow->deep (timm's native order);
    # only Encoder.channels/.strides (above) are reversed to deepest-first.
    assert [tuple(o.shape[1:]) for o in outs] == [(64, 16, 16), (128, 8, 8), (320, 4, 4), (512, 2, 2)]
    assert all(torch.isfinite(o).all() for o in outs)


# ---------------------------------------------------------------------------
# Strict weight loading: missing file / spurious key / missing key / allow_missing
# ---------------------------------------------------------------------------

def _make_fake_checkpoint(tmp_path: Path, arch: str = "resnet18") -> tuple[Path, str]:
    """A locally-constructed 'pretrained' checkpoint: an unpretrained
    resnet18's own state dict, saved to disk and sha256'd — self-consistent
    with whatever key layout build_encoder's own timm.create_model call
    produces for *arch*, with no network access and no dependency on
    guessing timm's internal naming."""
    baseline = build_encoder({"name": arch, "weights": "none"}, in_chans=3)
    path = tmp_path / "fake_encoder.pth"
    torch.save(baseline.module.state_dict(), path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha256


def test_missing_weights_file_raises():
    with pytest.raises(FileNotFoundError):
        build_encoder(
            {"name": "resnet18", "weights": "file:/no/such/checkpoint.pth", "sha256": "a" * 64},
            in_chans=3,
        )


def test_file_source_without_sha256_raises():
    path = "/tmp/whatever.pth"  # never reached — sha256 check happens first
    with pytest.raises(ValueError, match="sha256"):
        build_encoder({"name": "resnet18", "weights": f"file:{path}"}, in_chans=3)


def test_file_source_sha256_mismatch_raises(tmp_path):
    path, _real_sha = _make_fake_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        build_encoder({"name": "resnet18", "weights": f"file:{path}", "sha256": "0" * 64}, in_chans=3)


def test_spurious_key_raises(tmp_path):
    path, _sha = _make_fake_checkpoint(tmp_path)
    sd = torch.load(path)
    sd["totally_bogus_key"] = torch.zeros(1)
    bad_path = tmp_path / "bad_extra_key.pth"
    torch.save(sd, bad_path)
    bad_sha = hashlib.sha256(bad_path.read_bytes()).hexdigest()

    with pytest.raises(KeyError, match="totally_bogus_key"):
        build_encoder({"name": "resnet18", "weights": f"file:{bad_path}", "sha256": bad_sha}, in_chans=3)


def test_missing_key_raises_without_allow_missing(tmp_path):
    path, _sha = _make_fake_checkpoint(tmp_path)
    sd = torch.load(path)
    del sd["conv1.weight"]
    bad_path = tmp_path / "missing_key.pth"
    torch.save(sd, bad_path)
    bad_sha = hashlib.sha256(bad_path.read_bytes()).hexdigest()

    with pytest.raises(KeyError, match="conv1.weight"):
        build_encoder({"name": "resnet18", "weights": f"file:{bad_path}", "sha256": bad_sha}, in_chans=3)


def test_allow_missing_covers_declared_prefix(tmp_path):
    path, _sha = _make_fake_checkpoint(tmp_path)
    sd = torch.load(path)
    del sd["conv1.weight"]
    bad_path = tmp_path / "missing_key.pth"
    torch.save(sd, bad_path)
    bad_sha = hashlib.sha256(bad_path.read_bytes()).hexdigest()

    encoder = build_encoder(
        {"name": "resnet18", "weights": f"file:{bad_path}", "sha256": bad_sha, "allow_missing": ["conv1."]},
        in_chans=3,
    )
    assert encoder.weight_record.missing_keys == ["conv1.weight"]
    # Fair warning that this is a real gap, not silently patched over: the
    # allowed-missing param stays at its own random init, not the
    # checkpoint's value (there's nothing to copy it from).


# ---------------------------------------------------------------------------
# config_hash: weights is part of identity (plan §6.2 / §3.3)
# ---------------------------------------------------------------------------

def test_config_hash_changes_when_encoder_weights_changes(tiny_config_factory):
    cfg_a = tiny_config_factory()
    cfg_a["model"]["encoder"] = {"name": "pvt_v2_b2", "weights": "none"}
    validated_a = validate_config(copy.deepcopy(cfg_a))

    cfg_b = tiny_config_factory()
    cfg_b["model"]["encoder"] = {"name": "pvt_v2_b2", "weights": "timm:pvt_v2_b2.in1k"}
    validated_b = validate_config(copy.deepcopy(cfg_b))

    assert config_hash(validated_a) != config_hash(validated_b)
    # Re-validating the same config must be a no-op on the hash (determinism).
    assert config_hash(validated_a) == config_hash(validate_config(copy.deepcopy(validated_a)))


def test_encoder_config_file_source_requires_sha256_at_schema_level(tiny_config_factory):
    cfg = tiny_config_factory()
    cfg["model"]["encoder"] = {"name": "resnet18", "weights": "file:/tmp/whatever.pth"}
    with pytest.raises(pydantic.ValidationError, match="sha256"):
        validate_config(cfg)


def test_encoder_config_hf_source_requires_pinned_revision(tiny_config_factory):
    cfg = tiny_config_factory()
    cfg["model"]["encoder"] = {"name": "resnet18", "weights": "hf:some/repo"}
    with pytest.raises(pydantic.ValidationError, match="revision"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Input-stem adaptation: zero_init is function-preserving at initialisation
# ---------------------------------------------------------------------------

def test_zero_init_matches_pretrained_on_rgb_alone_for_any_extra(tmp_path):
    path, sha = _make_fake_checkpoint(tmp_path)

    baseline = build_encoder({"name": "resnet18", "weights": f"file:{path}", "sha256": sha}, in_chans=3)
    baseline.module.eval()

    widened = build_encoder(
        {"name": "resnet18", "weights": f"file:{path}", "sha256": sha, "in_chans_strategy": "zero_init"},
        in_chans=5,
    )
    widened.module.eval()

    torch.manual_seed(0)
    x_rgb = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        base_outs = baseline.module(x_rgb)

    for _ in range(2):  # "whatever values extra holds" — try two different draws
        extra = torch.randn(2, 2, 32, 32) * 10 - 3
        x5 = torch.cat([x_rgb, extra], dim=1)
        with torch.no_grad():
            wide_outs = widened.module(x5)
        for a, b in zip(base_outs, wide_outs):
            assert torch.equal(a, b)


def test_separate_stem_also_matches_pretrained_on_rgb_alone_at_init(tmp_path):
    """separate_stem's scratch branch is zero-initialised too, so it should
    have the same function-preserving property as zero_init at step 0 —
    the plan only names zero_init explicitly, but this follows from the
    same reasoning (§6.3) and is worth pinning down."""
    path, sha = _make_fake_checkpoint(tmp_path)
    baseline = build_encoder({"name": "resnet18", "weights": f"file:{path}", "sha256": sha}, in_chans=3)
    baseline.module.eval()

    separate = build_encoder(
        {"name": "resnet18", "weights": f"file:{path}", "sha256": sha, "in_chans_strategy": "separate_stem"},
        in_chans=5,
    )
    separate.module.eval()

    torch.manual_seed(1)
    x_rgb = torch.randn(2, 3, 32, 32)
    extra = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        base_outs = baseline.module(x_rgb)
        sep_outs = separate.module(torch.cat([x_rgb, extra], dim=1))
    for a, b in zip(base_outs, sep_outs):
        assert torch.allclose(a, b, atol=1e-6)


def test_grayscale_in_chans_one_uses_timm_builtin_adaptation(tmp_path):
    """in_chans=1 always goes through timm's own adapt_input_conv (sums the
    RGB kernels), independent of in_chans_strategy — plan §6.3."""
    path, sha = _make_fake_checkpoint(tmp_path)
    encoder = build_encoder(
        {"name": "resnet18", "weights": f"file:{path}", "sha256": sha, "in_chans_strategy": "zero_init"},
        in_chans=1,
    )
    encoder.module.eval()  # batch=1 + 32x32 collapses to 1x1 spatial by the last stage;
    # BatchNorm in train() mode rejects a single value per channel.
    x = torch.randn(1, 1, 32, 32)
    outs = encoder.module(x)
    assert all(torch.isfinite(o).all() for o in outs)
    stem = encoder.module.get_submodule(encoder.pretrained_cfg["first_conv"])
    assert stem.in_channels == 1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_resolve_pretrained_norm_stats_picks_timm_pretrained_cfg():
    mean, std = resolve_pretrained_norm_stats({"name": "pvt_v2_b2", "weights": "timm:pvt_v2_b2.in1k"})
    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]


def test_resolve_pretrained_norm_stats_falls_back_to_architecture_default():
    # weights: none still resolves stats from the architecture's own
    # registered default cfg (there is no "the weights this specific run
    # loaded" stats source when nothing was loaded).
    mean, std = resolve_pretrained_norm_stats({"name": "resnet18", "weights": "none"})
    assert len(mean) == 3 and len(std) == 3


def test_datamodule_overrides_dataset_norm_stats_when_pretrained(tiny_config_factory):
    """models/encoders.py's resolve_pretrained_norm_stats, wired into
    datasets/datamodule.py's BaseDataModule (plan §6.4): a model.encoder
    with normalization: pretrained overrides dataset.norm_mean/std before
    transforms are built, and records what it used."""
    from dissert.datasets import StandardSplitDataModule
    from dissert.training.determinism import get_recorded_manifest_extras, reset_recorded_nondeterminism

    reset_recorded_nondeterminism()
    cfg = tiny_config_factory()
    cfg["dataset"]["norm_mean"] = [0.0, 0.0, 0.0]  # would apply if not overridden
    cfg["dataset"]["norm_std"] = [1.0, 1.0, 1.0]
    cfg["model"]["encoder"] = {
        "name": "pvt_v2_b2", "weights": "none", "normalization": "pretrained",
    }

    dm = StandardSplitDataModule(cfg)

    expected_mean, expected_std = resolve_pretrained_norm_stats(cfg["model"]["encoder"])
    assert dm.resolved_norm_stats["mean"] == expected_mean
    assert dm.resolved_norm_stats["std"] == expected_std
    assert cfg["dataset"]["norm_mean"] == expected_mean
    assert cfg["dataset"]["norm_std"] == expected_std
    assert get_recorded_manifest_extras()["resolved_norm_stats"] == dm.resolved_norm_stats


def test_datamodule_leaves_dataset_norm_stats_alone_without_pretrained_encoder(tiny_config_factory):
    from dissert.datasets import StandardSplitDataModule

    cfg = tiny_config_factory()
    cfg["dataset"]["norm_mean"] = [0.1, 0.2, 0.3]
    cfg["dataset"]["norm_std"] = [0.4, 0.5, 0.6]

    dm = StandardSplitDataModule(cfg)

    assert dm.resolved_norm_stats is None
    assert cfg["dataset"]["norm_mean"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# End-to-end: both registered families build and run a forward pass through
# get_model() with the new structured encoder config.
# ---------------------------------------------------------------------------

def test_emcad_builds_and_forwards_with_structured_encoder_config():
    from dissert.models.registry import get_model

    model = get_model(
        name="emcad", encoder={"name": "pvt_v2_b2", "weights": "none"},
        in_channels=5, out_channels=1,
    )
    x = torch.randn(1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
    assert model.weight_record.source == "none"


def test_encoder_unet_builds_and_forwards():
    from dissert.models.registry import get_model

    model = get_model(
        name="encoder_unet", encoder={"name": "resnet18", "weights": "none"},
        in_channels=1, out_channels=2,
    )
    x = torch.randn(1, 1, 96, 96)
    out = model(x)
    assert out.shape == (1, 2, 96, 96)
