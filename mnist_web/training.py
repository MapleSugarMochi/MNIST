"""Reproducible MNIST training with explicit preprocessing and split provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from .artifacts import runtime_versions, write_manifest
from .preprocessing import PREPROCESSING_VERSION, normalize_mnist_images, preprocess_drawing


def build_model(seed: int = 42) -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", seed=seed),
            layers.RandomRotation(0.04, fill_mode="constant", seed=seed + 1),
            layers.RandomZoom(0.08, fill_mode="constant", seed=seed + 2),
        ],
        name="augmentation",
    )
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = augmentation(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="mnist_cnn")


def stratified_indices(
    labels: np.ndarray, seed: int, fraction: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    train, validation = np.array(train), np.array(validation)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def prepare_images(images: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normalized":
        return normalize_mnist_images(images)
    if mode == "drawing":
        return np.concatenate([preprocess_drawing(Image.fromarray(x)) for x in images])
    raise ValueError("Unknown preprocessing mode")


def train(
    output: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    preprocessing: str = "drawing",
    deterministic: bool = True,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or output.suffix != ".keras":
        raise ValueError("Use positive epochs/batch size and a .keras output")
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_ids, val_ids = stratified_indices(labels, seed)
    x = prepare_images(images, preprocessing)
    x_test = prepare_images(test_images, preprocessing)
    model = build_model(seed)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = "val_loss"
    history = model.fit(
        x[train_ids],
        labels[train_ids],
        validation_data=(x[val_ids], labels[val_ids]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor=monitor, patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                output, monitor=monitor, mode="min", save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=1, min_lr=1e-6),
        ],
    )
    best_model = keras.models.load_model(output, compile=False)
    # Evaluate the saved artifact, which is exactly the model later served.
    from .evaluation import classification_metrics, predict_batches

    metrics = classification_metrics(test_labels, predict_batches(best_model, x_test))
    model_id = f"mnist-{preprocessing}-seed{seed}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    manifest = write_manifest(output, model_id)
    split_path = output.with_suffix(".split.npz")
    np.savez_compressed(split_path, train_indices=train_ids, validation_indices=val_ids)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "deterministic": deterministic,
        "runtime": runtime_versions(),
        "hardware": [device.name for device in tf.config.list_physical_devices()],
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "selection_metric": monitor,
        "selection_mode": "min",
        "best_epoch": int(np.argmin(history.history[monitor])) + 1,
        "training_preprocessing": preprocessing,
        "inference_preprocessing": PREPROCESSING_VERSION,
        "split": {
            "method": "stratified",
            "validation_fraction": 0.1,
            "train_samples": len(train_ids),
            "validation_samples": len(val_ids),
            "validation_indices_sha256": hashlib.sha256(val_ids.tobytes()).hexdigest(),
        },
        "dataset_sha256": hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest(),
        "test_metrics": metrics,
        "manifest": manifest,
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {output}; held-out accuracy: {metrics['accuracy']:.4%}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.keras"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing", choices=("drawing", "normalized"), default="drawing")
    parser.add_argument("--non-deterministic", action="store_true")
    args = parser.parse_args()
    train(
        args.output.resolve(),
        args.epochs,
        args.batch_size,
        args.seed,
        args.preprocessing,
        not args.non_deterministic,
    )


if __name__ == "__main__":
    main()
