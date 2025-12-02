from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io

model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['image']  # base64格式
    image_data = data.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img  # 反色：黑底白字 → 白底黑字
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))

    return {"prediction": digit}

if __name__ == "__main__":
    app.run(debug=True)
