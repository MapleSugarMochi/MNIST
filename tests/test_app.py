import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from app import create_app


class FakeModel:
    def __call__(self, batch, training=False):
        assert batch.shape == (1, 28, 28, 1)
        assert training is False
        probabilities = np.full((1, 10), 0.04 / 7, dtype=np.float32)
        probabilities[0, 7] = 0.9
        probabilities[0, 1] = 0.04
        probabilities[0, 9] = 0.02
        return probabilities


def png_data_url(with_ink=True):
    image = Image.new("L", (280, 280), 0)
    if with_ink:
        ImageDraw.Draw(image).line((80, 40, 200, 230), fill=255, width=20)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_client():
    app = create_app(FakeModel())
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_and_health():
    client = make_client()
    assert client.get("/").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200
    assert client.get("/health").get_json() == {"status": "ok"}
    assert client.get("/collect").status_code == 200
    assert client.get("/static/drawing.js").status_code == 200
    assert client.get("/ready").get_json()["status"] == "ready"


def test_predict_returns_ranked_probabilities():
    response = make_client().post("/predict", json={"image": png_data_url()})
    body = response.get_json()
    assert response.status_code == 200
    assert body["prediction"] == 7
    assert body["confidence"] == 0.9
    assert [item["digit"] for item in body["top3"]] == [7, 1, 9]
    assert body["uncertain"] is None
    assert body["confidence_threshold"] is None


def test_missing_image_returns_json_400():
    response = make_client().post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_data_url_returns_json_400():
    response = make_client().post("/predict", json={"image": "not-an-image"})
    assert response.status_code == 400
    assert "PNG data URL" in response.get_json()["error"]


def test_blank_canvas_returns_json_400():
    response = make_client().post("/predict", json={"image": png_data_url(False)})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Draw a digit first."


def test_request_size_limit_returns_json_413():
    response = make_client().post(
        "/predict",
        data=b"x" * (2 * 1024 * 1024 + 1),
        content_type="application/json",
    )
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_missing_model_keeps_liveness_but_fails_readiness_and_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("MNIST_MODEL_PATH", str(tmp_path / "absent.keras"))
    client = create_app().test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    response = client.post("/predict", json={"image": png_data_url()})
    assert response.status_code == 503
    assert response.is_json
    assert "absent.keras" not in response.get_json()["error"]


def test_invalid_model_output_returns_json_503():
    class BadModel:
        def __call__(self, batch, training=False):
            return np.full((1, 10), np.nan)

    client = create_app(BadModel()).test_client()
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"image": png_data_url()}).status_code == 503


def test_model_is_warmed_once():
    class CountingModel(FakeModel):
        def __init__(self):
            self.calls = 0

        def __call__(self, batch, training=False):
            self.calls += 1
            return super().__call__(batch, training)

    model = CountingModel()
    client = create_app(model).test_client()
    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 200
    assert model.calls == 1
    client.post("/predict", json={"image": png_data_url()})
    assert model.calls == 2


def test_oversized_image_returns_json_400():
    image = Image.new("L", (2049, 1))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = make_client().post("/predict", json={"image": url})
    assert response.status_code == 400
    assert response.is_json
