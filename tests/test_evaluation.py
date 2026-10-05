import json

import numpy as np
import pytest

from mnist_web.evaluation import (
    choose_threshold,
    classification_metrics,
    load_canvas_corpus,
    require_independent_corpora,
)
from tests.test_app import png_data_url


def test_confusion_and_rejection_metrics_include_non_digits():
    labels = np.array([0, 0, 1, -1])
    probabilities = np.zeros((4, 10))
    probabilities[:, 0] = [0.9, 0.4, 0.1, 0.9]
    probabilities[:, 1] = 1 - probabilities[:, 0]
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"][0][:2] == [1, 1]
    assert metrics["per_class"][0]["recall"] == 0.5
    assert metrics["non_digit_samples"] == 1


def test_threshold_selection_handles_ties_and_insufficient_support():
    probabilities = np.zeros((4, 10))
    probabilities[:, 0] = [0.9, 0.9, 0.8, 0.6]
    probabilities[:, 1] = 1 - probabilities[:, 0]
    result = choose_threshold(np.array([0, -1, 0, -1]), probabilities, 0.9, 1)
    assert result["threshold"] is None  # May not accept just the correct half of a tied score.
    result = choose_threshold(np.array([0, 0, 0, -1]), probabilities, 0.95, 2)
    assert result["threshold"] == 0.8
    assert result["coverage"] == 0.75
    assert choose_threshold(np.zeros(4), probabilities, 0.95, 30)["threshold"] is None


def test_corpus_duplicates_and_writer_leakage_are_rejected(tmp_path):
    sample = {"label": 7, "writer_id": "writer-a", "image": png_data_url()}
    path = tmp_path / "canvas.json"
    path.write_text(
        json.dumps({"schema_version": 1, "source": "human_canvas", "samples": [sample, sample]})
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_canvas_corpus(path)
    with pytest.raises(ValueError, match="writers overlap"):
        require_independent_corpora(
            {"writer_ids": ["a"], "image_hashes": ["1"]},
            {"writer_ids": ["a"], "image_hashes": ["2"]},
        )
    with pytest.raises(ValueError, match="drawings overlap"):
        require_independent_corpora(
            {"writer_ids": ["a"], "image_hashes": ["1"]},
            {"writer_ids": ["b"], "image_hashes": ["1"]},
        )
