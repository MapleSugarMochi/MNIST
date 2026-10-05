import numpy as np
import pytest
from PIL import Image, ImageDraw

from preprocessing import EmptyDrawingError, preprocess_drawing


def test_blank_drawing_is_rejected():
    with pytest.raises(EmptyDrawingError, match="请先写一个数字"):
        preprocess_drawing(Image.new("L", (280, 280), 0))


def test_drawing_is_white_on_black_and_centered():
    image = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(image)
    draw.line((220, 40, 220, 220), fill=255, width=20)

    result = preprocess_drawing(image)

    assert result.shape == (1, 28, 28, 1)
    assert result.dtype == np.float32
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    assert float(result[0, :, :, 0].sum()) > 0
    assert float(result[0, 0, 0, 0]) == 0.0
    mass = result[0, :, :, 0]
    yy, xx = np.indices(mass.shape)
    assert abs(float((xx * mass).sum() / mass.sum()) - 13.5) <= 1
    assert abs(float((yy * mass).sum() / mass.sum()) - 13.5) <= 1


def test_transparent_hidden_pixels_are_not_visible_ink():
    with pytest.raises(EmptyDrawingError):
        preprocess_drawing(Image.new("RGBA", (280, 280), (255, 255, 255, 0)))


def test_position_and_canvas_scale_do_not_change_normalized_glyph():
    glyph = Image.new("L", (20, 100), 255)
    first = Image.new("L", (280, 280), 0)
    first.paste(glyph, (20, 100))
    second = Image.new("L", (560, 560), 0)
    second.paste(glyph.resize((40, 200)), (420, 40))
    np.testing.assert_allclose(preprocess_drawing(first), preprocess_drawing(second), atol=1 / 255)
