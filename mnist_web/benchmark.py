"""Warm model/API latency and concurrent request measurements (no throughput guarantee)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from .app import create_app
from .artifacts import model_identity, resolve_model_path, runtime_versions
from .preprocessing import preprocess_drawing


def summarize(times: list[float]) -> dict[str, float]:
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def benchmark(iterations: int = 30, workers: int = 4) -> dict:
    if iterations < 1 or workers < 1:
        raise ValueError("Iterations and workers must be positive")
    path = resolve_model_path()
    model = tf.keras.models.load_model(path, compile=False)
    image = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(image).line([(60, 60), (210, 60), (120, 230)], fill=255, width=20)
    batch = preprocess_drawing(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}
    report = {
        "model": model_identity(path),
        "runtime": runtime_versions(),
        "iterations": iterations,
        "concurrent_workers": workers,
    }
    for name, call in (
        ("direct", lambda: model(batch, training=False).numpy()),
        ("predict", lambda: model.predict(batch, verbose=0)),
    ):
        call()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            call()
            times.append((time.perf_counter() - start) * 1000)
        report[name] = summarize(times)
    app = create_app()
    start = time.perf_counter()
    assert app.test_client().get("/ready").status_code == 200
    report["cold_ready_ms"] = (time.perf_counter() - start) * 1000

    def request_time(_):
        with app.test_client() as client:
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            if response.status_code != 200 or response.get_json()["prediction"] != 7:
                raise RuntimeError("Benchmark inference failed")
            return (time.perf_counter() - start) * 1000

    report["api_sequential"] = summarize([request_time(i) for i in range(iterations)])
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        times = list(pool.map(request_time, range(iterations)))
    elapsed = time.perf_counter() - start
    report["api_concurrent"] = {
        **summarize(times),
        "requests_per_second": iterations / elapsed,
        "note": "In-process Flask clients, excluding HTTP network and WSGI overhead.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    report = benchmark(args.iterations, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("direct", "predict", "api_sequential", "api_concurrent")}
        )
    )


if __name__ == "__main__":
    main()
