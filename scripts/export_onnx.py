"""Export the pinned Keras model to ONNX for the static browser demo.

Requires the optional export tooling in requirements-export.txt. The exported
graph is verified against the Keras model on the full MNIST test set before
anything is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from mnist_web.artifacts import model_identity, resolve_model_path, sha256_file
from mnist_web.evaluation import predict_batches
from mnist_web.preprocessing import PREPROCESSING_VERSION, preprocess_drawing

OPSET = 17


def export(model_path: Path, output_dir: Path, tolerance: float = 1e-5) -> dict:
    import onnxruntime as ort
    import tf2onnx

    identity = model_identity(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    signature = [tf.TensorSpec([None, 28, 28, 1], tf.float32, name="image")]

    @tf.function(input_signature=signature)
    def serve(image):
        return {"probabilities": model(image, training=False)}

    proto, _ = tf2onnx.convert.from_function(serve, input_signature=signature, opset=OPSET)
    session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
    batch = np.concatenate([preprocess_drawing(Image.fromarray(x)) for x in images])
    expected = predict_batches(model, batch)
    actual = np.concatenate(
        [session.run(None, {input_name: batch[i : i + 500]})[0] for i in range(0, len(batch), 500)]
    )
    max_diff = float(np.abs(actual - expected).max())
    agreement = float((actual.argmax(1) == expected.argmax(1)).mean())
    if max_diff > tolerance or agreement != 1.0:
        raise RuntimeError(f"ONNX parity failed: max diff {max_diff}, agreement {agreement}")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    onnx_path.write_bytes(proto.SerializeToString())
    info = {
        "schema_version": 1,
        "model_id": identity["id"],
        "source_sha256": identity["sha256"],
        "onnx_sha256": sha256_file(onnx_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "opset": OPSET,
        "input": {"name": input_name, "shape": [None, 28, 28, 1], "dtype": "float32"},
        "output": session.get_outputs()[0].name,
        "parity": {
            "dataset": "MNIST test set, 10,000 images, serving preprocessing",
            "max_abs_probability_diff": max_diff,
            "argmax_agreement": agreement,
            "test_accuracy": float((actual.argmax(1) == labels).mean()),
        },
    }
    (output_dir / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=resolve_model_path())
    parser.add_argument("--output-dir", type=Path, default=Path("demo"))
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
