import os
import json
import base64
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────
IMG_SIZE   = 64
MODEL_PATH = "model.h5"

# ── Load class labels from the saved JSON ───────────────
# This guarantees Flask uses the EXACT same label order as training.
# {"A": 0, "B": 1, ...}  →  flip to  {0: "A", 1: "B", ...}
with open("class_indices.json") as f:
    _class_indices = json.load(f)
LABELS = {v: k for k, v in _class_indices.items()}
print(f"Labels loaded: {LABELS}")

# ── Load model ───────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded — input: {model.input_shape}, classes: {len(LABELS)}")

# ── MediaPipe ────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=True,   # each frame is independent — no stream
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

BOX_SIZE = 300  # must match what the frontend draws

# ── Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "image" not in data:
            return jsonify({"error": "No image field in request"}), 400

        # 1. Decode base64 → BGR image ────────────────────
        img_b64 = data["image"]
        _, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        h, w = img_bgr.shape[:2]

        # 2. Centre crop — same box the frontend draws ────
        cx, cy = w // 2, h // 2
        x1 = max(cx - BOX_SIZE // 2, 0)
        y1 = max(cy - BOX_SIZE // 2, 0)
        x2 = min(cx + BOX_SIZE // 2, w)
        y2 = min(cy + BOX_SIZE // 2, h)

        roi     = img_bgr[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # 3. MediaPipe hand detection on ROI ──────────────
        result       = hands.process(roi_rgb)
        hand_detected = result.multi_hand_landmarks is not None

        label           = None
        confidence      = None
        all_predictions = {}

        if hand_detected:
            # 4. Preprocess exactly as in training ────────
            # Training used: rescale=1./255, no flip, resize to (64,64)
            img_ml = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            img_ml = img_ml.astype("float32") / 255.0     # same as rescale=1./255
            img_ml = np.expand_dims(img_ml, axis=0)       # (1, 64, 64, 3)

            # 5. Predict ──────────────────────────────────
            preds      = model.predict(img_ml, verbose=0)
            class_id   = int(np.argmax(preds[0]))
            confidence = round(float(np.max(preds[0])), 4)
            label      = LABELS.get(class_id, f"class_{class_id}")

            all_predictions = {
                LABELS.get(i, f"class_{i}"): round(float(p), 4)
                for i, p in enumerate(preds[0])
            }

        return jsonify({
            "hand_detected":   hand_detected,
            "label":           label,
            "confidence":      confidence,
            "all_predictions": all_predictions,
            "box":             {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    except Exception as e:
        app.logger.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  MODEL_PATH,
        "labels": LABELS,
        "num_classes": len(LABELS)
    })


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)