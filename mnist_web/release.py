"""Build a portable, checksummed model bundle without publishing it."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .artifacts import model_identity, resolve_model_path, sha256_file


def build_bundle(model: Path, output: Path, evaluation: Path) -> Path:
    identity = model_identity(model)
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    if report["model"]["sha256"] != identity["sha256"]:
        raise ValueError("Evaluation report belongs to a different model")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing release artifacts")
    output.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (model, "model.keras"),
        (model.with_suffix(".manifest.json"), "model.manifest.json"),
        (evaluation, "evaluation.json"),
    ):
        shutil.copy2(source, output / name)
    manifest_path = output / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["filename"] = "model.keras"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata = model.with_suffix(".metadata.json")
    if metadata.is_file():
        shutil.copy2(metadata, output / "model.metadata.json")
    requirements = Path("requirements.lock")
    if requirements.is_file():
        shutil.copy2(requirements, output / requirements.name)
    entries = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=resolve_model_path())
    parser.add_argument("--output", type=Path, default=Path("dist/model-bundle"))
    parser.add_argument("--evaluation", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    print(build_bundle(args.model, args.output, args.evaluation))


if __name__ == "__main__":
    main()
