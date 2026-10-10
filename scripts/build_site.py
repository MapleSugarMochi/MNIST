"""Assemble the static browser demo into a deployable directory (default: _site/).

Requires `npm ci --prefix demo` for the vendored ONNX Runtime Web files.
Preview locally with: python -m http.server --directory _site
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
STATIC = ROOT / "mnist_web" / "static"
ORT_DIST = DEMO / "node_modules" / "onnxruntime-web" / "dist"
ORT_FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> Path:
    info = json.loads((DEMO / "model.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "mnist_web/assets/model.manifest.json").read_text("utf-8"))
    if info["source_sha256"] != manifest["sha256"]:
        raise SystemExit("demo/model.onnx is stale: re-run scripts/export_onnx.py")
    if info["onnx_sha256"] != sha256(DEMO / "model.onnx"):
        raise SystemExit("demo/model.onnx does not match demo/model.json")
    missing = [name for name in ORT_FILES if not (ORT_DIST / name).is_file()]
    if missing:
        raise SystemExit(f"Missing ONNX Runtime Web files {missing}; run: npm ci --prefix demo")

    if output.exists():
        shutil.rmtree(output)
    (output / "vendor").mkdir(parents=True)
    demo_files = (
        "index.html", "demo.css", "demo.js", "preprocessing.js", "model.onnx", "model.json",
    )
    for name in demo_files:
        shutil.copy2(DEMO / name, output / name)
    for name in ("drawing.js", "style.css", "favicon.svg"):
        shutil.copy2(STATIC / name, output / name)
    for name in ORT_FILES:
        shutil.copy2(ORT_DIST / name, output / "vendor" / name)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    print(build(parser.parse_args().output))


if __name__ == "__main__":
    main()
