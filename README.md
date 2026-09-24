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
├── vision.py          # Main vision pipeline & landmark feature extractor
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignored files (caches, virtual envs, model files)
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

## Landmark Extraction API

```python
from vision import extract_landmarks

# Returns a 1D numpy array of shape (63,) with dtype float32:
# [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
features = extract_landmarks(hand_landmarks)
```
