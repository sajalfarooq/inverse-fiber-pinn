"""Shared artifact formatting; no scientific models are hidden here."""
import json
import platform
from pathlib import Path
import importlib.metadata
import os
import re
import shutil
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def result_dir(project):
    """Resolve output without creating it; tagged runs cannot escape the project."""
    tag = os.environ.get("OPTICS_RUN_TAG")
    if tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
        raise ValueError("Run tag must be 1-64 letters, digits, underscores or hyphens")
    return Path(project) / "runs" / tag if tag else Path(project) / "results"


def setup(project):
    out = result_dir(project)
    if out.exists() and any(out.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = Path(project)/"archives"/f"{out.name}-{stamp}"
        root = Path(project).resolve()
        if not out.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):
            raise ValueError("Output and archive paths must stay inside the project")
        archive.parent.mkdir(parents=True,exist_ok=True)
        # Move the whole prior run so a failed new run cannot retain stale metrics.
        out.rename(archive)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    return out


def write_metrics(out, metrics):
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = {"data_source": "synthetic simulation; no experimental measurements",
               "python": platform.python_version(), "versions": versions, **metrics}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
