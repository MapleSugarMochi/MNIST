"""Separate MNIST, synthetic canvas, and human canvas evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from .app import decode_image_data_url, validate_probabilities
from .artifacts import model_identity, resolve_model_path, runtime_versions, sha256_file
from .preprocessing import PREPROCESSING_VERSION, normalize_mnist_images, preprocess_drawing


def predict_batches(model: Any, batch: np.ndarray, batch_size: int = 128) -> np.ndarray:
    if len(batch) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return np.concatenate(
        [
            validate_probabilities(
                model(batch[i : i + batch_size], training=False), min(batch_size, len(batch) - i)
            )
            for i in range(0, len(batch), batch_size)
        ]
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Class metrics plus calibration diagnostics; -1 denotes a non-digit."""
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    digit_mask = labels >= 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels[digit_mask], predicted[digit_mask]), 1)
    per_class = []
    for digit in range(10):
        true_positive = int(confusion[digit, digit])
        support = int(confusion[digit].sum())
        selected = int(confusion[:, digit].sum())
        precision = true_positive / selected if selected else 0.0
        recall = true_positive / support if support else 0.0
        per_class.append(
            {
                "digit": digit,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    bins = []
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (confidence >= low) & ((confidence < high) if index < 9 else (confidence <= high))
        count = int(mask.sum())
        if count:
            accuracy, mean_score = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += count / len(labels) * abs(accuracy - mean_score)
            bins.append(
                {
                    "lower": low,
                    "upper": high,
                    "count": count,
                    "accuracy": accuracy,
                    "mean_score": mean_score,
                }
            )
    digit_labels, digit_probs = labels[digit_mask], probabilities[digit_mask]
    return {
        "samples": len(labels),
        "digit_samples": int(digit_mask.sum()),
        "non_digit_samples": int((~digit_mask).sum()),
        "accuracy": float(correct[digit_mask].mean()) if digit_mask.any() else None,
        "macro_f1": float(np.mean([x["f1"] for x in per_class])),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "expected_calibration_error": ece,
        "reliability_bins": bins,
        "negative_log_likelihood": float(
            -np.log(
                np.maximum(digit_probs[np.arange(len(digit_labels)), digit_labels], 1e-7)
            ).mean()
        )
        if digit_mask.any()
        else None,
        "non_digit_mean_score": float(confidence[~digit_mask].mean())
        if (~digit_mask).any()
        else None,
    }


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_accuracy: float = 0.95,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Maximize retained validation samples at a requested empirical accuracy."""
    if not 0 < target_accuracy <= 1 or min_samples < 1:
        raise ValueError("Invalid threshold selection settings")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    # Aggregate ties: a threshold must retain every sample with the same score.
    order = np.argsort(-confidence, kind="stable")
    sorted_scores = confidence[order]
    counts = np.arange(1, len(order) + 1)
    accuracy = np.cumsum(correct[order]) / counts
    end_of_tie = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    eligible = np.flatnonzero(end_of_tie & (counts >= min_samples) & (accuracy >= target_accuracy))
    if len(eligible) == 0:
        return {
            "threshold": None,
            "accepted_samples": 0,
            "coverage": 0.0,
            "accepted_accuracy": None,
            "reason": "insufficient_validation_evidence",
        }
    index = int(eligible[-1])
    return {
        "threshold": float(sorted_scores[index]),
        "accepted_samples": int(counts[index]),
        "coverage": float(counts[index] / len(labels)),
        "accepted_accuracy": float(accuracy[index]),
        "reason": "selected_on_validation_only",
    }


def load_canvas_corpus(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("source") != "human_canvas" or corpus.get("schema_version") != 1:
        raise ValueError("Expected a version-1 human_canvas export from /collect")
    batches, labels, writers, hashes = [], [], set(), set()
    for sample in corpus["samples"]:
        label, writer = sample["label"], sample["writer_id"]
        if (
            type(label) is not int
            or not -1 <= label <= 9
            or not isinstance(writer, str)
            or not writer
        ):
            raise ValueError("Every sample needs label -1..9 and a nonempty anonymous writer_id")
        image = decode_image_data_url(sample["image"])
        image_hash = hashlib.sha256(
            str(image.size).encode() + image.convert("RGBA").tobytes()
        ).hexdigest()
        if image_hash in hashes:
            raise ValueError("Duplicate drawing in canvas corpus")
        hashes.add(image_hash)
        writers.add(writer)
        batches.append(preprocess_drawing(image))
        labels.append(label)
    if not batches:
        raise ValueError("Canvas corpus contains no samples")
    return (
        np.concatenate(batches),
        np.array(labels, dtype=np.int64),
        {
            "sha256": sha256_file(path),
            "writer_ids": sorted(writers),
            "image_hashes": sorted(hashes),
        },
    )


def require_independent_corpora(validation: dict[str, Any], test: dict[str, Any]) -> None:
    if set(validation["writer_ids"]) & set(test["writer_ids"]):
        raise ValueError("Validation/test writers overlap; split by writer, not by drawing")
    if set(validation["image_hashes"]) & set(test["image_hashes"]):
        raise ValueError("Validation/test drawings overlap")


def synthetic_canvas(images: np.ndarray, variant: str) -> np.ndarray:
    """Render held-out MNIST into a 280px canvas; this is not human evaluation."""
    batches = []
    for image in images:
        glyph = Image.fromarray(image).resize((168, 168), Image.Resampling.BILINEAR)
        position = (56, 56)
        if variant == "offset_small":
            glyph = glyph.resize((84, 84), Image.Resampling.BILINEAR)
            position = (170, 10)
        elif variant == "thick_tilt":
            glyph = glyph.filter(ImageFilter.MaxFilter(5)).rotate(
                12, resample=Image.Resampling.BILINEAR
            )
        elif variant == "thin":
            glyph = glyph.filter(ImageFilter.MinFilter(3))
        canvas = Image.new("L", (280, 280), 0)
        canvas.paste(glyph, position)
        batches.append(preprocess_drawing(canvas))
    return np.concatenate(batches)


def evaluate(
    model_path: Path,
    output: Path,
    limit: int | None = None,
    canvas_validation: Path | None = None,
    canvas_test: Path | None = None,
    calibration_output: Path | None = None,
    target_accuracy: float = 0.95,
    min_samples: int = 30,
) -> dict[str, Any]:
    identity = model_identity(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        images, labels = images[:limit], labels[:limit]
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": identity,
        "runtime": runtime_versions(),
        "mnist_test_selection": f"first {len(images)}",
        "mnist": classification_metrics(
            labels, predict_batches(model, normalize_mnist_images(images))
        ),
        "mnist_drawing_preprocessed": classification_metrics(
            labels,
            predict_batches(
                model, np.concatenate([preprocess_drawing(Image.fromarray(x)) for x in images])
            ),
        ),
        "synthetic_canvas": {},
        "human_canvas": {
            "status": "not_collected",
            "note": "Synthetic results are not human accuracy.",
        },
    }
    for variant in ("baseline", "offset_small", "thick_tilt", "thin"):
        report["synthetic_canvas"][variant] = classification_metrics(
            labels, predict_batches(model, synthetic_canvas(images, variant))
        )
    validation_identity = None
    calibration = None
    if canvas_validation:
        batch, truths, validation_identity = load_canvas_corpus(canvas_validation)
        probs = predict_batches(model, batch)
        calibration = {
            **choose_threshold(truths, probs, target_accuracy, min_samples),
            "source": "human_canvas_validation",
            "model_sha256": identity["sha256"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "validation": validation_identity,
            "target_accuracy": target_accuracy,
            "min_samples": min_samples,
            "note": "Empirical score threshold, not calibrated probability or OOD guarantee.",
        }
        report["human_canvas"] = {
            "validation": classification_metrics(truths, probs),
            "threshold_selection": calibration,
            "status": "validation_only",
        }
    if canvas_test:
        batch, truths, test_identity = load_canvas_corpus(canvas_test)
        if validation_identity:
            require_independent_corpora(validation_identity, test_identity)
        probs = predict_batches(model, batch)
        report["human_canvas"].update(
            status="test_evaluated",
            test=classification_metrics(truths, probs),
            test_dataset=test_identity,
        )
        if calibration and calibration["threshold"] is not None:
            accepted = probs.max(axis=1) >= calibration["threshold"]
            report["human_canvas"]["test_threshold_metrics"] = {
                "coverage": float(accepted.mean()),
                "accepted_accuracy": float(
                    (probs.argmax(axis=1)[accepted] == truths[accepted]).mean()
                )
                if accepted.any()
                else None,
                "non_digit_false_accepts": int(np.sum(accepted & (truths == -1))),
            }
    if calibration_output:
        if calibration is None:
            raise ValueError("Calibration requires an independent human canvas validation corpus")
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=resolve_model_path())
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--canvas-validation", type=Path)
    parser.add_argument("--canvas-test", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--min-samples", type=int, default=30)
    args = parser.parse_args()
    report = evaluate(
        args.model,
        args.output,
        args.limit,
        args.canvas_validation,
        args.canvas_test,
        args.calibration_output,
        args.target_accuracy,
        args.min_samples,
    )
    print(f"MNIST accuracy: {report['mnist']['accuracy']:.4%}; report: {args.output}")


if __name__ == "__main__":
    main()
