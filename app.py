"""Compatibility entry point: python app.py / waitress-serve app:app."""

from mnist_web.app import app, create_app, decode_image_data_url, main, resolve_model_path

__all__ = ["app", "create_app", "decode_image_data_url", "resolve_model_path"]

if __name__ == "__main__":
    main()
