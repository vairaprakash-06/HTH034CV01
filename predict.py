"""
predict.py - Standalone Inference Script for Hand Gesture Classification

Accepts a 63-coordinate floating-point array (21 3D landmarks: x, y, z),
loads the pre-trained scikit-learn SVM model (model.pkl),
and prints the predicted gesture/word.

Usage:
  1. Via stdin (recommended for child_process.spawn):
     echo [0.1, 0.2, ...] | python predict.py
  2. Via CLI argument:
     python predict.py "[0.1, 0.2, ...]"
"""

import json
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow / C++ logs if imported indirectly
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def load_model(model_path: str = MODEL_FILE):
    """
    Loads the trained SVM model and label encoder from disk.
    Supports dictionary bundle or tuple structure.
    """
    if not os.path.exists(model_path):
        return None, None

    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            model = bundle.get("model")
            label_encoder = bundle.get("label_encoder")
            return model, label_encoder
        elif isinstance(bundle, (tuple, list)):
            return bundle[0], bundle[1]
        else:
            return bundle, None
    except Exception as e:
        sys.stderr.write(f"[!] Error loading model from '{model_path}': {e}\n")
        return None, None


def parse_input_coordinates():
    """
    Reads coordinates from either command line argument or standard input.
    Expected format: JSON array of 63 numbers or JSON object with 'landmarks'/'features'.
    """
    raw_text = ""

    # Check CLI argument
    if len(sys.argv) > 1 and sys.argv[1].strip():
        raw_text = sys.argv[1].strip()
    else:
        # Read from stdin
        raw_text = sys.stdin.read().strip()

    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            coords = parsed.get("landmarks") or parsed.get("features") or parsed.get("coordinates")
            return coords
        elif isinstance(parsed, list):
            return parsed
        return None
    except Exception as e:
        sys.stderr.write(f"[!] JSON parse error: {e}\n")
        return None


def main():
    coords = parse_input_coordinates()

    if not coords or len(coords) != 63:
        sys.stderr.write(f"[!] Invalid input: Expected 63 coordinates, got {len(coords) if coords else 0}\n")
        # Return fallback response
        print(json.dumps({"prediction": "—", "confidence": 0.0, "error": "Invalid feature length"}))
        return

    # Convert to 2D numpy array of shape (1, 63)
    features = np.array(coords, dtype=np.float32).reshape(1, -1)

    # Load model
    model, label_encoder = load_model(MODEL_FILE)

    if model is None:
        # Graceful fallback when model.pkl is not yet trained
        sys.stderr.write("[!] 'model.pkl' not found. Please run 'python train.py' to train your classifier.\n")
        print(json.dumps({
            "prediction": "READY",
            "confidence": 0.50,
            "status": "model.pkl not yet created; run python train.py"
        }))
        return

    try:
        # Perform prediction
        pred_idx = model.predict(features)

        # Decode label
        if label_encoder is not None:
            predicted_word = str(label_encoder.inverse_transform(pred_idx)[0])
        else:
            predicted_word = str(pred_idx[0])

        # Calculate prediction confidence if supported
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            except Exception:
                pass

        # Print JSON output for Express server
        output = {
            "prediction": predicted_word,
            "confidence": round(confidence, 4)
        }
        print(json.dumps(output))

    except Exception as e:
        sys.stderr.write(f"[!] Inference error: {e}\n")
        print(json.dumps({"prediction": "—", "confidence": 0.0, "error": str(e)}))


if __name__ == "__main__":
    main()
