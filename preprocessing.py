"""Compatibility imports for the packaged preprocessing implementation."""

from mnist_web.preprocessing import (
    EmptyDrawingError,
    normalize_mnist_images,
    preprocess_drawing,
)

__all__ = ["EmptyDrawingError", "normalize_mnist_images", "preprocess_drawing"]
