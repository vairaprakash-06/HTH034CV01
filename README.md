# Sign Language Bridge

Real-time sign language interpreter vision pipeline built with Python, OpenCV, and MediaPipe.

## Features
- **Real-Time Hand Landmark Tracking**: Tracks 21 3D hand keypoints per hand from live webcam video.
- **63-Feature Extraction**: Helper function `extract_landmarks()` flattens `(x, y, z)` coordinates into a 1D NumPy array `(63,)` ready for machine learning / classification models.
- **Dual MediaPipe Architecture**: Compatible with both modern MediaPipe Tasks API (`HandLandmarker`) and legacy `mp.solutions.hands`.
- **Performance Optimized**: Windows DirectShow (`CAP_DSHOW`) backend, minimal frame buffering, and temporal video tracking for high FPS (>45 FPS on CPU).
- **Clean VS Code Integration**: Seamless exit handling via `'q'` or window close button with proper camera hardware release.

## Project Structure
```text
sign-language-bridge/
├── server.js          # Express API server (port 3000, CORS, /predict endpoint)
├── predict.py         # Lightweight Python inference worker (SVM model loader)
├── index.html         # Web kiosk interface (HTML5 video, canvas overlay)
├── main.js            # Frontend logic (MediaPipe Tasks Vision CDN, fetch API)
├── style.css          # Kiosk design system (Public service counter aesthetic)
├── package.json       # Project scripts and dependencies (Vite, Express, CORS)
├── collect_data.py    # Dataset collection tool (logs landmarks to CSV)
├── train.py           # SVM classifier training script (scikit-learn)
├── vision.py          # Python vision pipeline & landmark extractor
├── requirements.txt   # Python dependencies
├── landmarks.csv      # Generated dataset (63 coordinates + label)
├── model.pkl          # Trained SVM model bundle (model + LabelEncoder)
├── .gitignore         # Ignored files (caches, node_modules, dist)
└── README.md          # Documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vairaprakash-06/sign-language-bridge.git
   cd sign-language-bridge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the real-time webcam vision pipeline:
```bash
python vision.py
```

### CLI Arguments
| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--width` | `640` | Video feed width |
| `--height` | `480` | Video feed height |
| `--max-hands` | `2` | Maximum hands to detect simultaneously |

Example:
```bash
python vision.py --camera 0 --width 640 --height 480 --max-hands 2
```

Press **`q`** or **`ESC`** in the video window to quit.

## Dataset Collection

Run `collect_data.py` to record labeled hand landmark data into `landmarks.csv`:

```bash
python collect_data.py
```

### Controls & Keybindings
| Action | Key | Description |
|---|---|---|
| **Record Letter / Digit** | Press `'A'` - `'Z'` or `'0'` - `'9'` | Extracts 63 landmarks and appends a row with that label to `landmarks.csv` |
| **Record Active Label** | `SPACEBAR` | Appends a row for the currently selected active label |
| **Quit** | `'q'` or `ESC` | Saves dataset progress and exits cleanly |

### CLI Arguments
| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `landmarks.csv` | Output CSV dataset file path |
| `-l`, `--label` | `A` | Default initial active label |
| `-c`, `--camera` | `0` | Camera device index |
| `--width` | `640` | Video feed width |
| `--height` | `480` | Video feed height |
| `--max-hands` | `1` | Max hands to detect |

### CSV Format (`landmarks.csv`)
Each row consists of **64 columns**:
1. `label`: The gesture class label (e.g., `'A'`, `'B'`, `'HELLO'`).
2. `x0, y0, z0, ..., x20, y20, z20`: 63 floating-point coordinates for the 21 3D hand keypoints normalized to `[0, 1]` or camera space.

## Model Training

Once you've collected samples in `landmarks.csv`, train an SVM classifier using `train.py`:

```bash
python train.py
```

### CLI Arguments
| Argument | Default | Description |
|---|---|---|
| `-d`, `--data` | `landmarks.csv` | Path to dataset CSV file |
| `-o`, `--output` | `model.pkl` | Path to save trained model bundle |
| `-k`, `--kernel` | `rbf` | SVM kernel (`rbf`, `linear`, `poly`) |
| `-C`, `--c-val` | `1.0` | SVM regularization parameter |
| `--test-size` | `0.2` | Fraction of dataset reserved for testing (80/20 split) |
| `--seed` | `42` | Random seed for reproducible splits |

Example:
```bash
python train.py --data landmarks.csv --kernel rbf --output model.pkl
```

### Loading the Trained Model in Backend Scripts

```python
import joblib

# Load the model bundle
bundle = joblib.load("model.pkl")
model = bundle["model"]
label_encoder = bundle["label_encoder"]

# Predict gesture from 63 landmark features:
prediction_encoded = model.predict([features_63])
gesture = label_encoder.inverse_transform(prediction_encoded)[0]

# Optional: Get prediction confidence
probabilities = model.predict_proba([features_63])[0]
confidence = max(probabilities)
print(f"Predicted Gesture: {gesture} ({confidence * 100:.1f}%)")
```

## Web Application (Vite Kiosk Terminal)

A browser-based real-time interpreter web application styled as a modern public service-counter kiosk.

### Features
- **HTML5 `<video>` & `<canvas>`**: Captures webcam stream with live skeletal landmark overlay.
- **MediaPipe Tasks Vision (CDN)**: Extracts 21 3D hand landmarks in real-time in the browser.
- **63-Feature Payload**: Formats `[x0, y0, z0, ..., x20, y20, z20]` and dispatches to `POST http://localhost:3000/predict`.
- **Public Kiosk UX**: Hero prediction card, confidence gauge, communication transcript sentence builder, and Text-to-Speech (TTS) announcement.

### Running the Web App
```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

### Expected Backend API (`http://localhost:3000/predict`)
The web application sends:
```json
{
  "landmarks": [0.521, 0.632, -0.012, "... 63 floats total ..."]
}
```

And expects a JSON response such as:
```json
{
  "prediction": "A",
  "confidence": 0.98
}
```

## Express Backend API (`server.js`)

A high-performance Express.js microservice bridging the web kiosk to Python's scikit-learn SVM inference pipeline.

### Starting the Server
```bash
npm run server
# or: node server.js
```

The server starts on `http://localhost:3000` with CORS enabled.

### Endpoint: `POST /predict`
- **Request Body**:
  ```json
  {
    "landmarks": [0.48, 0.62, -0.01, "... 63 floats total ..."]
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "HELLO",
    "confidence": 0.9167
  }
  ```

### Architecture
1. Frontend makes a fast HTTP POST to Express `POST /predict`.
2. `server.js` uses `child_process.spawn` to pipe the coordinates into `predict.py` via `stdin`.
3. `predict.py` loads `model.pkl`, executes `model.predict()`, and returns the translated label.
4. `server.js` formats and returns the response as JSON to the kiosk client.




