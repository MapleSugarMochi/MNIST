import json
from pathlib import Path

import numpy as np
import pytest

from mnist_web.artifacts import ASSETS_DIR, model_identity, sha256_file
from mnist_web.preprocessing import PREPROCESSING_VERSION, preprocess_drawing

DEMO = Path(__file__).resolve().parents[1] / "demo"


def test_browser_model_is_exported_from_the_pinned_model():
    info = json.loads((DEMO / "model.json").read_text(encoding="utf-8"))
    identity = model_identity(ASSETS_DIR / "model.keras")
    assert info["model_id"] == identity["id"]
    assert info["source_sha256"] == identity["sha256"]
    assert info["onnx_sha256"] == sha256_file(DEMO / "model.onnx")
    assert info["preprocessing_version"] == PREPROCESSING_VERSION
    assert info["parity"]["argmax_agreement"] == 1.0


def test_onnx_model_matches_keras_on_canvas_drawing():
    ort = pytest.importorskip("onnxruntime")
    import tensorflow as tf
    from PIL import Image, ImageDraw

    image = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(image).line([(60, 60), (210, 60), (120, 230)], fill=255, width=20)
    batch = preprocess_drawing(image)
    keras_model = tf.keras.models.load_model(ASSETS_DIR / "model.keras", compile=False)
    session = ort.InferenceSession(str(DEMO / "model.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: batch})[0]
    expected = np.asarray(keras_model(batch, training=False))
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert int(actual.argmax()) == 7
