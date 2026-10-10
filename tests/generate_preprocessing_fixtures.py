"""Regenerate tests/fixtures/preprocessing_parity.json from the Python pipeline.

The fixture pins the server preprocessing output for canvas-like drawings and
MNIST test digits so tests/frontend.test.cjs can check the browser port
(demo/preprocessing.js) pixel for pixel. Run: python -m tests.generate_preprocessing_fixtures
"""

import base64
import json
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mnist_web.preprocessing import PREPROCESSING_VERSION, preprocess_drawing

OUTPUT = Path(__file__).parent / "fixtures" / "preprocessing_parity.json"


def encode(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, np.uint8).tobytes()
    return base64.b64encode(zlib.compress(raw, 9)).decode()


def canvases(rng: random.Random):
    fixed = [
        [(60, 60), (210, 60), (120, 230)],  # seven
        [(140, 30), (140, 250)],  # tall, narrow one
        [(40, 140), (240, 150)],  # wide, flat stroke
        [(250, 10), (270, 30), (255, 60)],  # small and offset
    ]
    for points, width in zip(fixed, (20, 20, 20, 6), strict=True):
        image = Image.new("L", (280, 280), 0)
        ImageDraw.Draw(image).line(points, fill=255, width=width, joint="curve")
        yield image
    for _ in range(36):
        image = Image.new("L", (280, 280), 0)
        draw = ImageDraw.Draw(image)
        for _ in range(rng.randint(1, 3)):
            count = rng.randint(2, 6)
            points = [(rng.randint(10, 270), rng.randint(10, 270)) for _ in range(count)]
            draw.line(points, fill=rng.randint(120, 255), width=rng.randint(3, 30), joint="curve")
        yield image


def main() -> None:
    import tensorflow as tf

    samples = []
    for image in canvases(random.Random(0)):
        gray = np.asarray(image)
        expected = np.rint(preprocess_drawing(image)[0, ..., 0] * 255)
        samples.append({"source": "canvas", "width": 280, "height": 280,
                        "gray": encode(gray), "expected": encode(expected)})
    (_, _), (images, _) = tf.keras.datasets.mnist.load_data()
    for image in images[:300]:
        expected = np.rint(preprocess_drawing(Image.fromarray(image))[0, ..., 0] * 255)
        samples.append({"source": "mnist", "width": 28, "height": 28,
                        "gray": encode(image), "expected": encode(expected)})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"preprocessing_version": PREPROCESSING_VERSION,
                                  "samples": samples}) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
