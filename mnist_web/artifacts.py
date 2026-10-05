"""Model identity, integrity verification, and portable manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .preprocessing import PREPROCESSING_VERSION

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("tensorflow", "keras", "numpy", "Pillow", "Flask")
        },
    }


def resolve_model_path() -> Path:
    configured = os.getenv("MNIST_MODEL_PATH")
    path = Path(configured).expanduser().resolve() if configured else ASSETS_DIR / "model.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return path


def model_identity(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Model manifest uses an incompatible preprocessing version")
    expected = os.getenv("MNIST_MODEL_SHA256") or manifest.get("sha256")
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError("Model SHA-256 does not match its manifest/configuration")
    return {
        "id": manifest["model_id"],
        "sha256": actual,
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def write_manifest(path: Path, model_id: str) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "filename": path.name,
        "sha256": sha256_file(path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_shape": [None, 28, 28, 1],
        "output_classes": list(range(10)),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
