"""Run from outside the checkout against a wheel installed into a fresh environment."""

import base64
import io

from PIL import Image, ImageDraw

from mnist_web.app import create_app

client = create_app().test_client()
assert client.get("/").status_code == 200
assert client.get("/collect").status_code == 200
for name in ("style.css", "drawing.js", "script.js", "collect.js", "favicon.svg"):
    assert client.get(f"/static/{name}").status_code == 200
ready = client.get("/ready")
assert ready.status_code == 200, ready.get_json()
image = Image.new("L", (280, 280), 0)
ImageDraw.Draw(image).line([(60, 60), (210, 60), (120, 230)], fill=255, width=20)
buffer = io.BytesIO()
image.save(buffer, "PNG")
response = client.post(
    "/predict",
    json={
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
    },
)
assert response.status_code == 200
assert response.get_json()["prediction"] == 7
print("Installed wheel: templates, static assets, hash verification and real inference passed.")
