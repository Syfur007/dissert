"""
tests/_pvt_parity_probe.py — helper script for
tests/test_encoders.py::test_pvt_v2_b0_parity_vendored_vs_timm_port.

Run as a standalone subprocess (never imported directly) so it gets a fresh
Python process: ``python _pvt_parity_probe.py <path-to-vendored-pvtv2.py>``.
Prints one line of JSON: ``{"shapes": [...], "diffs": [...]}`` — per-stage
output shapes and max-abs-difference between the vendored PVTv2 (loaded from
the given source file, recovered from git history — see the parity test's
docstring for why it's no longer in the tree) and timm's native
``pvt_v2_b0``, after transplanting the vendored model's (randomly
initialised) weights into timm's port via a mechanical key remap.

Must run in its own process: importing the vendored module registers
``pvt_v2_b0``..``pvt_v2_b5`` into timm's *global* model registry under the
same names as timm's own built-in versions (``@register_model``), silently
overwriting them for the rest of that process — precisely the collision
documented in models/encoders.py's module docstring, and the reason
``models/pvtv2.py`` could not simply be kept in the active import path.
Building the real timm reference model *before* importing the vendored
module (below) sidesteps it for this one comparison; nothing else in this
process ever calls ``timm.create_model`` for a pvt_v2_* name afterwards.
"""
import importlib.util
import json
import sys

import torch
import timm


def _remap_old_to_timm_keys(old_sd: dict) -> dict:
    remapped = {}
    for k, v in old_sd.items():
        if k.startswith("patch_embed1."):
            new_k = k.replace("patch_embed1.", "patch_embed.", 1)
        elif k.startswith("patch_embed"):
            stage = int(k[len("patch_embed")]) - 1
            new_k = f"stages_{stage}.downsample." + k.split(".", 1)[1]
        elif k.startswith("block"):
            head, rest = k.split(".", 1)
            stage = int(head[len("block"):]) - 1
            new_k = f"stages_{stage}.blocks.{rest}".replace(
                "mlp.dwconv.dwconv.", "mlp.dwconv."
            )
        elif k.startswith("norm") and k.split(".")[0][4:].isdigit():
            head, rest = k.split(".", 1)
            stage = int(head[4:]) - 1
            new_k = f"stages_{stage}.norm.{rest}"
        else:
            raise KeyError(f"Unexpected vendored PVTv2 key during remap: {k}")
        remapped[new_k] = v
    return remapped


def main(vendored_path: str) -> None:
    # Real timm architecture FIRST — see module docstring.
    new = timm.create_model(
        "pvt_v2_b0", features_only=True, out_indices=(0, 1, 2, 3), pretrained=False
    )

    spec = importlib.util.spec_from_file_location("_vendored_pvtv2", vendored_path)
    vendored = importlib.util.module_from_spec(spec)
    # timm's @register_model decorator looks itself up via
    # sys.modules[fn.__module__] — needs the module registered before exec.
    sys.modules[spec.name] = vendored
    spec.loader.exec_module(vendored)
    old = vendored.pvt_v2_b0()

    new.load_state_dict(_remap_old_to_timm_keys(old.state_dict()), strict=True)

    old.eval()
    new.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        old_outs = old(x)
        new_outs = new(x)

    result = {
        "shapes_old": [list(o.shape) for o in old_outs],
        "shapes_new": [list(o.shape) for o in new_outs],
        "diffs": [(o1 - o2).abs().max().item() for o1, o2 in zip(old_outs, new_outs)],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
