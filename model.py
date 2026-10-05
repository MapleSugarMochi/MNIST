"""Compatibility training entry point: python model.py."""

from mnist_web.training import build_model, main, train

__all__ = ["build_model", "train"]

if __name__ == "__main__":
    main()
