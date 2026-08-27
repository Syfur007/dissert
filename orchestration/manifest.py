"""
Run manifest: the single artifact that explains how a run's checkpoints/logs
came to exist — resolved config, config hash, code version, environment,
hardware, timing — written to artifacts/runs/<run_id>/manifest.json.

Every field-gathering helper here is best-effort: a manifest with a missing
optional field (no nvidia-smi on a CPU box, no psutil installed) is far more
useful than a run that crashes because manifest-building itself failed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runid import config_hash as _config_hash


def git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    """The current commit hash, or None outside a git repo / on any error.
    Public wrapper around the same lookup :class:`RunManifest` uses
    internally — exposed so utils.checkpoint.CheckpointManager can embed the
    commit in a saved checkpoint without duplicating the subprocess call."""
    return _git_info(repo_root)["commit"]


def _git_info(repo_root: Optional[str] = None) -> Dict[str, Any]:
    cwd = repo_root or os.getcwd()
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode()
        dirty = bool(status.strip())
    except Exception:
        pass
    return {"commit": commit, "dirty": dirty}


def _env_hash() -> Optional[str]:
    """SHA1 over the sorted installed-package list (name==version), as a
    cheap proxy for "which environment produced this run" without shelling
    out to `pip freeze`. None if package metadata can't be enumerated."""
    try:
        import importlib.metadata as importlib_metadata
        pkgs = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib_metadata.distributions()
            if dist.metadata and dist.metadata.get("Name")
        )
        return hashlib.sha1("\n".join(pkgs).encode("utf-8")).hexdigest()
    except Exception:
        return None


def _hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"uname": list(platform.uname())}

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name()
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        if smi:
            info["nvidia_smi"] = smi
    except Exception:
        pass

    try:
        import psutil
        info["host_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass

    return info


class RunManifest:
    """Mutable manifest for one run.

    Usage::

        manifest = build_manifest(run_id, resolved_config, seed, fold)
        manifest.start()
        try:
            ... train ...
            manifest.finish(status="done")
        except Exception as exc:
            manifest.finish(status="failed", error=str(exc))
        manifest.save(f"artifacts/runs/{run_id}/manifest.json")
    """

    def __init__(
        self,
        run_id: str,
        resolved_config: Dict[str, Any],
        seed: int,
        fold: Optional[int] = None,
        repo_root: Optional[str] = None,
    ):
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "config_hash": _config_hash(resolved_config),
            "resolved_config": resolved_config,
            "seed": seed,
            "fold": fold,
            "git": _git_info(repo_root),
            "env_hash": _env_hash(),
            "hardware": _hardware_info(),
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "gpu_hours": None,
            # training/determinism.py appends here whenever torch's
            # deterministic-algorithms guard actually trips on a
            # non-deterministic op, so a run can be *reported* non-
            # reproducible instead of silently assumed reproducible
            # (Phase 6's Mamba kernel path is the first model expected to
            # ever populate this).
            "nondeterministic_ops": [],
        }

    def start(self) -> "RunManifest":
        self.data["start_time"] = datetime.now(timezone.utc).isoformat()
        self.data["status"] = "running"
        return self

    def record_nondeterminism(self, note: str) -> None:
        self.data["nondeterministic_ops"].append(note)

    def record(self, key: str, value: Any) -> None:
        """Escape hatch for phase-specific extras (e.g. Phase 5's achieved
        width-match, Phase 6's active scan implementation) without every
        future phase needing to edit this class."""
        self.data[key] = value

    def finish(self, status: str = "done", error: Optional[str] = None) -> "RunManifest":
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self.data["status"] = status
        if error is not None:
            self.data["error"] = error
        if self.data["start_time"] is not None:
            start = datetime.fromisoformat(self.data["start_time"])
            end = datetime.fromisoformat(self.data["end_time"])
            wall_hours = (end - start).total_seconds() / 3600.0
            # Wall-clock proxy, not device-weighted GPU-second accounting —
            # Phase 10's profiling module supersedes this with the real
            # figure; kept under the same key name per the plan so callers
            # don't need to know which phase produced it, just that it may
            # be a coarse proxy until profiling lands.
            self.data["gpu_hours"] = wall_hours if self.data["hardware"].get("cuda_available") else None
        return self

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)


def build_manifest(
    run_id: str,
    resolved_config: Dict[str, Any],
    seed: int,
    fold: Optional[int] = None,
    repo_root: Optional[str] = None,
) -> RunManifest:
    """Construct a RunManifest for one run. Caller drives start()/finish()/save()
    around the actual training/eval call — this function itself performs no I/O."""
    return RunManifest(run_id, resolved_config, seed, fold, repo_root)
