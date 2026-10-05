import json

import pytest

from mnist_web.artifacts import ASSETS_DIR, model_identity, sha256_file
from mnist_web.release import build_bundle


def test_bundle_cannot_mix_models_and_checksums_cover_every_file(tmp_path):
    model = ASSETS_DIR / "model.keras"
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"model": {"sha256": "wrong"}}))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="different model"):
        build_bundle(model, output, report)
    report.write_text(json.dumps({"model": model_identity(model)}))
    build_bundle(model, output, report)
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert sha256_file(output / name) == digest
    assert model_identity(output / "model.keras") == model_identity(model)
    with pytest.raises(ValueError, match="empty output"):
        build_bundle(model, output, report)
