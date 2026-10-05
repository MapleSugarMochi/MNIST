import json

import numpy as np
import tensorflow as tf

from mnist_web.app import create_app
from mnist_web.artifacts import ASSETS_DIR, model_identity
from mnist_web.evaluation import predict_batches
from mnist_web.preprocessing import normalize_mnist_images, preprocess_drawing
from mnist_web.training import train
from tests.test_app import png_data_url


def test_pinned_model_loads_and_real_api_predicts_seven(monkeypatch):
    monkeypatch.delenv("MNIST_MODEL_PATH", raising=False)
    monkeypatch.delenv("MNIST_MODEL_SHA256", raising=False)
    monkeypatch.delenv("MNIST_CALIBRATION_PATH", raising=False)
    identity = model_identity(ASSETS_DIR / "model.keras")
    client = create_app().test_client()
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.get_json()["model"] == identity
    # A browser-like hand-drawn seven, not an injected classifier.
    import base64
    import io

    from PIL import Image, ImageDraw

    image = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(image).line([(60, 60), (210, 60), (120, 230)], fill=255, width=20)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = client.post("/predict", json={"image": url})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == 7
    assert response.get_json()["model"] == identity
    assert sum(x["confidence"] for x in response.get_json()["top3"]) <= 1.000002


def test_sha256_mismatch_and_wrong_calibration_fail_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("MNIST_MODEL_SHA256", "0" * 64)
    assert create_app().test_client().get("/ready").status_code == 503
    monkeypatch.delenv("MNIST_MODEL_SHA256")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"model_sha256": "0" * 64}))
    monkeypatch.setenv("MNIST_CALIBRATION_PATH", str(calibration))
    assert create_app().test_client().get("/ready").status_code == 503


def test_real_model_mnist_and_drawing_pipeline_regression():
    from PIL import Image

    (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
    images, labels = images[:1000], labels[:1000]
    model = tf.keras.models.load_model(ASSETS_DIR / "model.keras", compile=False)
    normal = predict_batches(model, normalize_mnist_images(images)).argmax(axis=1)
    drawing = predict_batches(
        model, np.concatenate([preprocess_drawing(Image.fromarray(image)) for image in images])
    ).argmax(axis=1)
    assert np.mean(normal == labels) >= 0.97
    assert np.mean(drawing == labels) >= 0.97


def test_training_saves_reproducible_artifact_and_explicit_split(monkeypatch, tmp_path):
    original = tf.keras.datasets.mnist.load_data()
    miniature = (
        (original[0][0][:256], original[0][1][:256]),
        (original[1][0][:64], original[1][1][:64]),
    )
    monkeypatch.setattr(tf.keras.datasets.mnist, "load_data", lambda: miniature)
    first = tmp_path / "first.keras"
    metadata = train(first, epochs=2, batch_size=64, seed=42)
    assert metadata["selection_metric"] == "val_loss"
    assert metadata["best_epoch"] in (1, 2)
    assert metadata["training_preprocessing"] == "drawing"
    assert metadata["manifest"]["sha256"] == model_identity(first)["sha256"]
    with np.load(first.with_suffix(".split.npz")) as split:
        assert not set(split["train_indices"]) & set(split["validation_indices"])
        assert len(split["train_indices"]) + len(split["validation_indices"]) == 256
    second = tmp_path / "second.keras"
    train(second, epochs=2, batch_size=64, seed=42)
    a = tf.keras.models.load_model(first, compile=False)
    b = tf.keras.models.load_model(second, compile=False)
    for weights_a, weights_b in zip(a.get_weights(), b.get_weights(), strict=True):
        np.testing.assert_allclose(weights_a, weights_b, atol=1e-6)


def test_validation_threshold_is_used_only_for_matching_model(monkeypatch, tmp_path):
    from mnist_web.preprocessing import PREPROCESSING_VERSION

    identity = model_identity(ASSETS_DIR / "model.keras")
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "source": "human_canvas_validation",
                "model_sha256": identity["sha256"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "threshold": 0.99,
            }
        )
    )
    monkeypatch.setenv("MNIST_CALIBRATION_PATH", str(path))
    response = create_app().test_client().post("/predict", json={"image": png_data_url()})
    assert response.status_code == 200
    body = response.get_json()
    assert body["uncertain"] == (body["confidence"] < 0.99)
