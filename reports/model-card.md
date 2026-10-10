# Model card: `mnist-drawing-seed42-20261010T103405Z`

| Field | Value |
| --- | --- |
| Task | Single handwritten digit classification, classes 0–9 |
| Architecture | `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(10, softmax)` with training-only translation, rotation, and zoom augmentation |
| Input | `float32[N, 28, 28, 1]` in `[0, 1]`, white ink on black, produced by preprocessing `white-ink-crop20-mass-center28-v1` |
| Keras artifact | `mnist_web/assets/model.keras`, SHA-256 `45a77cdde6359b9aa32ac8142cd3c5f19264a55eb3aa1ab23602ad759e423f0c` |
| Browser artifact | `demo/model.onnx` (opset 17), exported from the Keras artifact by `scripts/export_onnx.py` |
| Evaluated | 2026-10-10 |

## Training

```bash
python model.py --output artifacts/model.keras --epochs 20 --batch-size 128 --seed 42
```

- **Data:** MNIST training set. A stratified split holds out 10% per class: 54,001 training and 5,999 validation images. The MNIST array hash (image and label bytes) is `edc0d1098aa497be14ace79ddd5171e5a3dcf9c4d196c015389203eb3ba1f59b`.
- **Preprocessing:** training, validation, and test images go through the same crop, resize, and center-of-mass pipeline that the service applies to canvas drawings.
- **Optimization:** Adam with sparse categorical cross-entropy. Early stopping (patience 3), checkpointing, and learning-rate halving all monitor `val_loss`.
- **Outcome:** training stopped after 16 epochs and restored epoch 13 (`val_loss` 0.0339, `val_accuracy` 98.97%).
- **Environment:** Linux x86-64, 2 vCPU, CPU only, Python 3.13.16, with dependencies installed from `requirements.lock` by hash (TensorFlow 2.20.0, Keras 3.12.0, NumPy 2.3.4, Pillow 12.0.0).
- **Reproducibility:** TensorFlow deterministic ops are enabled. A second run in the same environment produced bit-identical weights and training history. The `.keras` file hash differs between runs because the archive stores a save timestamp. Bitwise identity across hardware is not guaranteed.

The full record, including the per-epoch history and split hash, is in `mnist_web/assets/model.metadata.json`.

## Evaluation

All rows use the full MNIST test set of 10,000 images. Synthetic canvases render MNIST test digits onto a black 280×280 canvas and pass them through the serving pipeline. They are controlled transformations, not human handwriting.

| Input condition | Accuracy | Macro F1 |
| --- | ---: | ---: |
| MNIST, normalization only | 99.23% | 0.9922 |
| MNIST, serving preprocessing | 99.24% | 0.9923 |
| Synthetic canvas, regular | 99.24% | 0.9923 |
| Synthetic canvas, small and offset | 99.26% | 0.9925 |
| Synthetic canvas, thick strokes and 12° tilt | 98.58% | 0.9857 |
| Synthetic canvas, thin strokes | 99.18% | 0.9917 |

With serving preprocessing, the expected calibration error is 0.0012 and the negative log-likelihood is 0.0235. The weakest classes by F1 are 9 (0.986, recall 97.7%) and 5 (0.988). The most frequent confusions are 9 → 4 (9 images), 5 → 3 (6), and 9 → 7 (5). Per-class metrics, confusion matrices, and reliability bins for every condition are in `reports/evaluation.json`.

### Browser parity

- **Model:** the ONNX export was checked against the Keras model on all 10,000 preprocessed test images. The maximum absolute probability difference is 1.3 × 10⁻⁶, and the argmax agreement is 100%.
- **Preprocessing:** the browser port (`demo/preprocessing.js`) reproduces the Python pipeline exactly on 340 fixtures: 40 synthetic canvas drawings and 300 MNIST digits. This check runs in CI.

## Performance

These measurements come from `reports/benchmark.json`. Each figure is the median of 30 warm calls on the same 2-vCPU Linux environment.

| Operation | Median | p95 |
| --- | ---: | ---: |
| Direct model call | 6.38 ms | 10.36 ms |
| Flask test client, sequential | 7.49 ms | 8.82 ms |
| Flask test client, 4 concurrent workers | 35.06 ms | 45.40 ms |

The concurrent run sustained about 103 requests per second. Inference is serialized behind a lock, which explains the higher concurrent latency. The measurements are in-process and exclude HTTP and WSGI overhead.

## Limitations

- **No human-handwriting evaluation yet.** The `/collect` page and `--canvas-validation` / `--canvas-test` evaluation support it, but no human corpus has been collected. Real drawings may differ from MNIST in stroke width, slant, and style.
- **No confidence threshold by default.** A threshold is enabled only from a human-canvas validation file that matches this model's hash.
- **Scores are not guarantees.** A softmax score does not guarantee correctness, and the model does not reliably reject non-digits, multiple digits, or scribbles.
- **Single digits only.** The model is not intended for multi-digit recognition or general OCR.
