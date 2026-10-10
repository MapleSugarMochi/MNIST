"""Shared image preprocessing for training and browser inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGE_SIZE = 28
GLYPH_SIZE = 20
INK_THRESHOLD = 20
MIN_INK_PIXELS = 12
PREPROCESSING_VERSION = "white-ink-crop20-mass-center28-v1"


class EmptyDrawingError(ValueError):
    """Raised when an image does not contain enough visible ink."""


def normalize_mnist_images(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images, dtype=np.float32) / 255.0
    if array.ndim in (2, 3):
        array = array[..., np.newaxis]
    return array


def _translate_without_wrap(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.zeros_like(array)
    height, width = array.shape
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    result[target_y0:target_y1, target_x0:target_x1] = array[
        source_y0:source_y1, source_x0:source_x1
    ]
    return result


def preprocess_drawing(image: Image.Image) -> np.ndarray:
    """Crop, scale, center, and normalize a black-canvas/white-ink drawing."""
    # Composite transparency onto the same black background as the browser canvas.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "black"), rgba)
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    ink_mask = grayscale > INK_THRESHOLD
    if int(np.count_nonzero(ink_mask)) < MIN_INK_PIXELS:
        raise EmptyDrawingError("Draw a digit first.")

    rows, columns = np.where(ink_mask)
    cropped = grayscale[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    height, width = cropped.shape
    scale = min(GLYPH_SIZE / width, GLYPH_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = Image.fromarray(cropped).resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )

    centered = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    centered[top : top + resized_height, left : left + resized_width] = np.asarray(resized)

    mass = centered.astype(np.float64)
    total = mass.sum()
    if total <= 0:
        raise EmptyDrawingError("The stroke is too small. Please draw again.")
    y_coordinates, x_coordinates = np.indices(centered.shape)
    center_x = float((x_coordinates * mass).sum() / total)
    center_y = float((y_coordinates * mass).sum() / total)
    target = (IMAGE_SIZE - 1) / 2
    centered = _translate_without_wrap(centered, round(target - center_x), round(target - center_y))

    return normalize_mnist_images(centered)[np.newaxis, ...]
