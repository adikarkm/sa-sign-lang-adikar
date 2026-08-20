import os
import base64
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landmark_model.keras")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

CLASS_NAMES = ["african beer", "hello", "how", "how are you", "okay", "think"]

# Initialize Models
keras_model = None
detector = None

def init_models():
    global keras_model, detector
    if keras_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        print("TensorFlow model loaded successfully.")

    if detector is None:
        if not os.path.exists(LANDMARKER_PATH):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, LANDMARKER_PATH)

        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe Landmarker loaded successfully.")

init_models()

def decode_base64_image(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Error decoding image:", e)
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image payload provided"}), 400

    image = decode_base64_image(data["image"])
    if image is None:
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    h, w, _ = image.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = detector.detect(mp_image)

    response_data = {
        "success": True,
        "hand_detected": False,
        "label": "No Hand Detected",
        "confidence": 0.0,
        "predictions": {name: 0.0 for name in CLASS_NAMES},
        "landmarks": []
    }

    if results.hand_landmarks:
        live_coordinates = []
        draw_landmarks_list = []

        for hand in results.hand_landmarks:
            hand_pts = []
            for lm in hand:
                hand_pts.append({"x": lm.x, "y": lm.y, "z": lm.z})
            draw_landmarks_list.append(hand_pts)

            wrist = hand[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            for lm in hand:
                live_coordinates.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

        if len(results.hand_landmarks) == 1:
            live_coordinates.extend([0] * 63)

        if len(live_coordinates) == 126:
            input_data = np.array([live_coordinates])
            preds = keras_model.predict(input_data, verbose=0)[0]

            preds_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
            best_idx = int(np.argmax(preds))
            best_conf = float(preds[best_idx] * 100)

            response_data["hand_detected"] = True
            response_data["label"] = CLASS_NAMES[best_idx]
            response_data["confidence"] = round(best_conf, 1)
            response_data["predictions"] = {k: round(v * 100, 1) for k, v in preds_dict.items()}
            response_data["landmarks"] = draw_landmarks_list

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
