# Sign Language Bridge

Real-time sign language interpreter vision pipeline and interactive accessibility kiosk built with Python, OpenCV, MediaPipe Tasks Vision, and JavaScript.

## Features
- **Dual-Hand Landmark Tracking (42 Keypoints)**: Tracks both hands simultaneously (21 3D points per hand = 42 points total) with distinct visual colors (Cyber Cyan for Hand 1, Sunset Amber for Hand 2).
- **Real-Time Gesture Calculation Engine**: High-speed geometric & kinematic recognizer that calculates both **two-hand gestures** (Namaste 🙏, Heart ❤️, Double Thumbs Up 👍👍, Double Peace ✌️✌️, Stop/Cross 🙅, Open Book 📖, Clap 👏, Together 🤝, Welcome 👐) and **single-hand gestures** (Thumbs Up 👍, Peace ✌️, OK 👌, I Love You 🤟, Rock On 🤘, Call Me 🤙, Pointing 👉, Hello ✋, Fist ✊, Pinch 🤏).
- **Interactive Accessibility Chatbox**: Real-time sign language conversation feed featuring:
  - Live staging buffer with hold-to-send circular/linear progress bar.
  - User Signer message bubbles with gesture emoji, confidence score, and hand count badge.
  - Automated AI Counter Assistant responses providing contextual replies.
  - Text-to-Speech (TTS) readout powered by the Web Speech API.
  - Manual text input, message editing, and one-click chat transcript export.
  - Quick-access gesture palette chips.
- **126-Coordinate Feature Payload**: Flattens `(x, y, z)` coordinates for up to 42 points into 126 floating-point values ready for machine learning inference and dataset recording.
- **Hybrid Inference & Telemetry**: Seamless bridge to Express (`POST /predict`) and Python `predict.py` with zero-latency client-side calculation fallback.
- **DirectShow Camera Acceleration**: Low-latency video capture (`CAP_DSHOW` on Windows) for high FPS (>45 FPS on CPU).

## Project Structure
```text
sign-language-bridge/
├── server.js          # Express API server (port 3000, CORS, /predict endpoint)
├── predict.py         # Lightweight Python inference worker (SVM model loader + 42-pt heuristic)
├── index.html         # Web kiosk interface (HTML5 video, canvas overlay, Chatbox)
├── main.js            # Frontend logic (MediaPipe Tasks Vision CDN, 42-pt tracking, Chatbox)
├── style.css          # Kiosk design system (Public service counter aesthetic)
├── package.json       # Project scripts and dependencies (Vite, Express, CORS)
├── collect_data.py    # Dataset collection tool (logs 42-pt / 21-pt landmarks to CSV)
├── train.py           # SVM classifier training script (scikit-learn, supports 63 & 126 feats)
├── vision.py          # Python vision pipeline & dual-hand landmark extractor
├── requirements.txt   # Python dependencies
├── landmarks.csv      # Generated dataset (coordinates + label)
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

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

## Web Application (Vite Kiosk & Real-Time Chatbox)

### Running the Web Kiosk
Start the Vite development server:
```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

### Starting the Backend Prediction API (Optional)
```bash
npm run server
# or: node server.js
```
The Express server runs on `http://localhost:3000` with CORS enabled.

### Web Kiosk Features & Controls
| Feature | Description |
|---|---|
| **Dual-Hand Tracking** | Detects up to 2 hands (42 points). Shows `0/42`, `21/42`, or `42/42` points in real time. |
| **Real-Time Gesture Calculation** | Instantly calculates both two-hand and single-hand signs every frame. |
| **Accessibility Chatbox** | Automatically transcribes recognized signs as messages in a conversational chat feed. |
| **Hold-To-Send** | Holding a gesture steady for 1.1s automatically commits it to the chatbox. |
| **AI Assistant Responses** | Counter #04 Assistant automatically replies to recognized signs. |
| **Text-to-Speech (TTS)** | Reads out recognized signs and assistant replies aloud using browser speech synthesis. |
| **Chat Transcript Export** | Download the entire chat log as a timestamped `.txt` file with one click. |
| **Gesture Chips** | Clickable chips along the bottom of the chatbox for testing and quick message insertion. |

---

## Python Vision Pipeline (`vision.py`)

Run the real-time webcam vision pipeline with dual-hand (42 points) tracking:
```bash
python vision.py --max-hands 2
```

### CLI Arguments
| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--width` | `640` | Video feed width |
| `--height` | `480` | Video feed height |
| `--max-hands` | `2` | Maximum hands to detect simultaneously (2 hands = 42 points) |

Press **`q`** or **`ESC`** in the video window to quit.

---

## Dataset Collection (`collect_data.py`)

Run `collect_data.py` to record labeled dual-hand (42 points / 126 coordinates) or single-hand landmark data into `landmarks.csv`:

```bash
python collect_data.py --max-hands 2
```

### Controls & Keybindings
| Action | Key | Description |
|---|---|---|
| **Record Letter / Digit** | Press `'A'` - `'Z'` or `'0'` - `'9'` | Extracts landmarks and appends a row with that label to `landmarks.csv` |
| **Record Active Label** | `SPACEBAR` | Appends a row for the currently selected active label |
| **Quit** | `'q'` or `ESC` | Saves dataset progress and exits cleanly |

### CSV Format (`landmarks.csv`)
Each row consists of:
1. `label`: Gesture class label (e.g. `'NAMASTE'`, `'HEART'`, `'HELLO'`).
2. `x0, y0, z0, ..., x41, y41, z41`: 126 coordinates for 42 hand keypoints (or 63 coordinates for 21 keypoints).

---

## Model Training (`train.py`)

Train an SVM classifier on collected samples in `landmarks.csv` (supports both 63 and 126 features automatically):

```bash
python train.py --data landmarks.csv --output model.pkl
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
