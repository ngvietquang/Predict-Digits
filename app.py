from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
from NeuralNetwork import NeuralNetwork
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load model
model = NeuralNetwork()
model.load_model("Predict-Digits/train.npz")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]

   
    image_data = base64.b64decode(data.split(",")[1])
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Threshold
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # Find bounding box
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize giữ tỉ lệ
    size = 20
    h_, w_ = digit.shape
    if h_ > w_:
        new_h = size
        new_w = int(w_ * size / h_)
    else:
        new_w = size
        new_h = int(h_ * size / w_)

    digit = cv2.resize(digit, (new_w, new_h))

    # Pad thành 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = digit

    # Normalize
    x_input = padded.astype(np.float32).reshape(-1) / 255.0

    # Predict
    probs = model.forward_propagation(x_input)
    pred = int(np.argmax(probs))
    conf = float(np.max(probs) * 100)

    return jsonify({
        "prediction": pred,
        "confidence": round(conf, 2)
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)
