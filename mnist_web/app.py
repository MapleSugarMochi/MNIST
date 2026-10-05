"""Flask endpoints with verified model identity and warmed inference."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from .artifacts import model_identity, resolve_model_path
from .preprocessing import PREPROCESSING_VERSION, preprocess_drawing

DATA_URL_PATTERN = re.compile(r"^data:image/png;base64,(?P<data>[A-Za-z0-9+/=]+)$")
MAX_IMAGE_SIDE = 2048


def decode_image_data_url(data_url: Any) -> Image.Image:
    if not isinstance(data_url, str):
        raise ValueError("字段 image 必须是 PNG Data URL。")
    match = DATA_URL_PATTERN.fullmatch(data_url)
    if not match:
        raise ValueError("仅支持 base64 编码的 PNG Data URL。")
    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片的 base64 数据无效。") from exc
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("图片内容不是 PNG 格式。")
            if max(source.size) > MAX_IMAGE_SIDE:
                raise ValueError("图片尺寸过大。")
            source.load()
            return source.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解析图片。") from exc


def validate_probabilities(output: Any, size: int = 1) -> np.ndarray:
    probabilities = np.asarray(output, dtype=np.float32)
    if (
        probabilities.shape != (size, 10)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Model output must contain ten normalized, finite probabilities")
    return probabilities


class InferenceService:
    def __init__(self, model: Any | None, logger: logging.Logger):
        self.model = model
        self.logger = logger
        self.lock = threading.Lock()
        self.warmed = False
        self.identity = {"id": "injected", "preprocessing_version": PREPROCESSING_VERSION}
        self.threshold: float | None = None

    def _initialize(self) -> None:
        if self.warmed:
            return
        if self.model is None:
            path = resolve_model_path()
            identity = model_identity(path)
            model = tf.keras.models.load_model(path, compile=False)
            validate_probabilities(model(np.zeros((1, 28, 28, 1), np.float32), training=False))
            calibration_path = os.getenv("MNIST_CALIBRATION_PATH")
            threshold = None
            if calibration_path:
                calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
                if (
                    calibration.get("model_sha256") != identity["sha256"]
                    or calibration.get("preprocessing_version") != PREPROCESSING_VERSION
                    or calibration.get("source") != "human_canvas_validation"
                ):
                    raise ValueError("Calibration does not match the model and drawing pipeline")
                threshold = calibration.get("threshold")
                if threshold is not None and (
                    not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1
                ):
                    raise ValueError("Invalid confidence threshold")
            self.model, self.identity, self.threshold = model, identity, threshold
            self.logger.info("Loaded model path=%s identity=%s", path, json.dumps(identity))
        else:
            validate_probabilities(self.model(np.zeros((1, 28, 28, 1), np.float32), training=False))
        self.warmed = True

    def ready(self) -> dict[str, Any]:
        with self.lock:
            self._initialize()
            return dict(self.identity)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self.lock:
            self._initialize()
            return validate_probabilities(self.model(batch, training=False))[0]


def create_app(model: Any | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    service = InferenceService(model, app.logger)
    app.extensions["mnist_inference"] = service

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/collect")
    def collect():
        return render_template("collect.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}, 200

    @app.get("/ready")
    def ready():
        try:
            return jsonify(status="ready", model=service.ready()), 200
        except Exception:
            app.logger.exception("Model readiness failed")
            return jsonify(status="unavailable", error="模型暂不可用。"), 503

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "image" not in payload:
            return jsonify(error="请求体必须包含 image 字段。"), 400
        try:
            batch = preprocess_drawing(decode_image_data_url(payload["image"]))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        try:
            probabilities = service.predict(batch)
        except Exception:
            app.logger.exception("Model inference failed")
            return jsonify(error="识别服务暂不可用，请稍后重试。"), 503
        indices = np.argsort(probabilities)[::-1][:3]
        confidence = float(probabilities[indices[0]])
        return jsonify(
            prediction=int(indices[0]),
            confidence=round(confidence, 6),
            top3=[
                {"digit": int(i), "confidence": round(float(probabilities[i]), 6)} for i in indices
            ],
            uncertain=confidence < service.threshold if service.threshold is not None else None,
            confidence_threshold=service.threshold,
            model=service.identity,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_: RequestEntityTooLarge):
        return jsonify(error="请求体不能超过 2 MB。"), 413

    return app


app = create_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )


if __name__ == "__main__":
    main()
