"""
collect_data.py - Sign Language Dataset Collection Tool

Features:
- Captures live webcam feed using OpenCV.
- Detects 21 3D hand landmarks in real-time using MediaPipe Hands.
- Flattens the 21 3D landmarks into 63 floating-point numbers (x, y, z per landmark).
- Listens for keyboard inputs to record gestures (e.g., press 'A'-'Z' or '0'-'9' to record that label,
  or press SPACE to record for the current active label).
- Appends recorded samples to `landmarks.csv` with format: [label, x0, y0, z0, ..., x20, y20, z20].
- Renders an interactive real-time visual HUD showing:
  * Current session sample count (prominently displayed)
  * Breakdown of samples collected per label in this session
  * Total samples in the CSV file
  * Real-time hand tracking status (DETECTED / NO HAND)
  * Visual flash/notification confirming each successful sample capture.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Suppress low-level MediaPipe / TensorFlow C++ logging in terminal
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

# Import hand detector and landmark extraction pipeline
try:
    from vision import SignLanguageHandDetector, extract_landmarks, MODEL_PATH, ensure_model_downloaded
except ImportError:
    # Graceful fallback in case vision.py is not in current working directory
    import urllib.request
    import mediapipe as mp

    MODEL_FILENAME = "hand_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def ensure_model_downloaded(model_path: str = MODEL_PATH) -> str:
        if not os.path.exists(model_path):
            print(f"[*] Downloading MediaPipe model to '{model_path}'...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("[*] Model download complete.")
        return model_path

    def extract_landmarks(hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(21 * 3, dtype=np.float32)
        landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
        coords = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        if len(coords) != 63:
            features = np.zeros(63, dtype=np.float32)
            features[:min(len(coords), 63)] = coords[:63]
            return features
        return np.array(coords, dtype=np.float32)

    class SignLanguageHandDetector:
        def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self.use_tasks_api = not (hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"))
            self._timestamp_ms = 0

            if self.use_tasks_api:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections

                model_file = ensure_model_downloaded(MODEL_PATH)
                base_options = python.BaseOptions(model_asset_path=model_file)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.drawing_utils = drawing_utils
                self.connections = HandLandmarksConnections.HAND_CONNECTIONS
            else:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)
                self.detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )

        def process_frame(self, frame_bgr: np.ndarray):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.use_tasks_api:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self._timestamp_ms += 33
                result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
                return result.hand_landmarks
            else:
                frame_rgb.flags.writeable = False
                results = self.detector.process(frame_rgb)
                frame_rgb.flags.writeable = True
                return results.multi_hand_landmarks if results.multi_hand_landmarks else []

        def draw_landmarks_on_frame(self, frame_bgr: np.ndarray, hand_landmarks_list) -> None:
            if not hand_landmarks_list:
                return
            for hand_landmarks in hand_landmarks_list:
                if self.use_tasks_api:
                    self.drawing_utils.draw_landmarks(frame_bgr, hand_landmarks, self.connections)
                else:
                    if self.mp_drawing_styles:
                        landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
                        connection_style = self.mp_drawing_styles.get_default_hand_connections_style()
                    else:
                        landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        connection_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    self.mp_drawing.draw_landmarks(
                        frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, landmark_style, connection_style
                    )

        def close(self):
            if hasattr(self.detector, "close"):
                self.detector.close()


def generate_header() -> List[str]:
    """Generates the standard 64-column CSV header: label, x0, y0, z0, ..., x20, y20, z20."""
    header = ["label"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def init_csv_file(csv_path: str) -> int:
    """
    Initializes the dataset CSV file with headers if it doesn't already exist.
    Returns the count of existing sample rows in the file.
    """
    existing_samples = 0
    file_exists = os.path.isfile(csv_path)

    if file_exists and os.path.getsize(csv_path) > 0:
        # Count existing data rows (excluding header)
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 0:
                    # If first row has 'label', data rows count is len(rows) - 1
                    if rows[0] and rows[0][0].lower() == "label":
                        existing_samples = max(0, len(rows) - 1)
                    else:
                        existing_samples = len(rows)
        except Exception as e:
            print(f"[!] Warning reading existing CSV '{csv_path}': {e}")
    else:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(csv_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(generate_header())
        print(f"[*] Initialized new dataset file with header: '{csv_path}'")

    return existing_samples


def append_sample_to_csv(csv_path: str, label: str, landmarks: List[float]) -> bool:
    """
    Appends a new sample row to the CSV file.
    Row structure: [label, x0, y0, z0, ..., x20, y20, z20] (64 values total).
    """
    if len(landmarks) != 63:
        print(f"[!] Error: Expected 63 landmark coordinates, got {len(landmarks)}.")
        return False

    # Format numbers to 6 decimal places for cleanliness and storage efficiency
    formatted_landmarks = [round(float(v), 6) for v in landmarks]
    row = [label] + formatted_landmarks

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"[!] Error writing sample to '{csv_path}': {e}")
        return False


def draw_hud(
    frame: np.ndarray,
    session_count: int,
    total_file_samples: int,
    active_label: str,
    hand_detected: bool,
    label_counts: Dict[str, int],
    feedback_msg: str,
    feedback_is_error: bool,
    feedback_timer: float,
) -> None:
    """
    Draws a modern, semi-transparent HUD overlay on the live feed showing:
    - Session sample counter
    - Hand detection indicator
    - Active label & per-label sample breakdown
    - Feedback / flash notifications
    - Interactive hotkey instructions
    """
    h, w, _ = frame.shape

    # 1. Top HUD Banner (Semi-transparent dark overlay)
    banner_height = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 2. Prominent Session Samples Counter
    cv2.putText(
        frame,
        f"SESSION SAMPLES: {session_count}",
        (18, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (0, 255, 128),  # Vibrant Emerald Green
        2,
        cv2.LINE_AA,
    )

    # 3. All-time / File sample count badge
    all_time_text = f"(Total in file: {total_file_samples + session_count})"
    cv2.putText(
        frame,
        all_time_text,
        (380, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # 4. Hand Detection Status Badge
    if hand_detected:
        status_text = "[HAND DETECTED - READY]"
        status_color = (0, 230, 100)  # Bright Green
    else:
        status_text = "[NO HAND DETECTED]"
        status_color = (0, 120, 255)  # Orange-Red Alert

    cv2.putText(
        frame,
        status_text,
        (18, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # 5. Active label and quick sample breakdown
    breakdown_items = [f"{lbl}:{cnt}" for lbl, cnt in sorted(label_counts.items())[-6:]]
    breakdown_text = f"Active: '{active_label}'"
    if breakdown_items:
        breakdown_text += f" | Recent: {', '.join(breakdown_items)}"

    cv2.putText(
        frame,
        breakdown_text,
        (18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 6. Bottom Banner: Controls / Keybindings
    footer_height = 36
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_height), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(footer_overlay, 0.75, frame, 0.25, 0, frame)

    controls_text = "Press 'A'-'Z' / '0'-'9' to Record | SPACE: Record Active | 'q'/ESC: Quit"
    cv2.putText(
        frame,
        controls_text,
        (18, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 200, 255),
        1,
        cv2.LINE_AA,
    )

    # 7. Action / Notification Banner (Flashes when sample is recorded or error occurs)
    now = time.time()
    if feedback_msg and (now - feedback_timer < 1.4):
        msg_color = (0, 60, 255) if feedback_is_error else (0, 255, 128)
        bg_color = (40, 20, 20) if feedback_is_error else (20, 45, 20)

        # Draw floating notification box
        box_y1 = banner_height + 10
        box_y2 = box_y1 + 38
        cv2.rectangle(frame, (14, box_y1), (w - 14, box_y2), bg_color, -1)
        cv2.rectangle(frame, (14, box_y1), (w - 14, box_y2), msg_color, 2)
        cv2.putText(
            frame,
            feedback_msg,
            (24, box_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            msg_color,
            2,
            cv2.LINE_AA,
        )

        # Border flash effect on capture
        if not feedback_is_error:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 128), 3)


def run_data_collection(
    csv_path: str = "landmarks.csv",
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    default_label: str = "A",
    max_hands: int = 1,
) -> None:
    """
    Main webcam data collection loop.
    Captures frames, extracts 21 3D landmarks (63 floats), listens for keypresses,
    and logs samples to CSV with live visual feedback.
    """
    window_name = "Sign Language Dataset Collector - collect_data.py"

    # Initialize CSV file and read existing sample count
    total_file_samples = init_csv_file(csv_path)

    # Optimize camera backend for Windows (CAP_DSHOW provides instant startup & low latency)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera with index {camera_index}.")
        print("[!] Please check webcam connection or try a different index (--camera 1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 65)
    print("  SIGN LANGUAGE DATASET COLLECTOR INITIALIZED")
    print("=" * 65)
    print(f"- Output CSV File       : {os.path.abspath(csv_path)}")
    print(f"- Existing File Samples : {total_file_samples}")
    print(f"- Camera Resolution     : {actual_w}x{actual_h}")
    print(f"- Initial Active Label  : '{default_label.upper()}'")
    print("- Keybindings:")
    print("    * Press 'A'-'Z' or '0'-'9' : Record sample with that character label")
    print("    * Press SPACEBAR           : Record sample with current active label")
    print("    * Press 'q' or ESC         : Save and exit")
    print("=" * 65 + "\n")

    detector = SignLanguageHandDetector(max_num_hands=max_hands)

    session_samples = 0
    label_counts: Dict[str, int] = {}
    active_label = default_label.upper()

    feedback_msg = ""
    feedback_is_error = False
    feedback_timer = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame for hand landmarks
            hands_landmarks = detector.process_frame(frame)
            hand_detected = bool(hands_landmarks and len(hands_landmarks) > 0)

            # Draw landmarks on frame
            if hand_detected:
                detector.draw_landmarks_on_frame(frame, hands_landmarks)

            # Render HUD overlay on video frame
            draw_hud(
                frame=frame,
                session_count=session_samples,
                total_file_samples=total_file_samples,
                active_label=active_label,
                hand_detected=hand_detected,
                label_counts=label_counts,
                feedback_msg=feedback_msg,
                feedback_is_error=feedback_is_error,
                feedback_timer=feedback_timer,
            )

            cv2.imshow(window_name, frame)

            # Listen for keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Exit on 'q', 'Q', or ESC
            if key in (ord("q"), ord("Q"), 27):
                break

            # Handle window close button ('X')
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            target_label: Optional[str] = None

            # 1. Direct letter keys ('a'-'z' or 'A'-'Z')
            if ord("a") <= key <= ord("z"):
                target_label = chr(key).upper()
            elif ord("A") <= key <= ord("Z"):
                target_label = chr(key)
            # 2. Number keys ('0'-'9')
            elif ord("0") <= key <= ord("9"):
                target_label = chr(key)
            # 3. Spacebar: Record for current active label
            elif key == ord(" "):
                target_label = active_label

            if target_label is not None:
                active_label = target_label

                if not hand_detected:
                    feedback_msg = f"[!] No hand detected! Position hand in frame to record '{target_label}'."
                    feedback_is_error = True
                    feedback_timer = time.time()
                    print(f"[*] Attempted to record '{target_label}', but no hand was detected.")
                else:
                    # Extract 21 3D landmarks as a flattened list of 63 floats
                    primary_hand = hands_landmarks[0]
                    features_1d = extract_landmarks(primary_hand)
                    features_list = [float(v) for v in features_1d.tolist()]

                    # Append row to landmarks.csv
                    success = append_sample_to_csv(csv_path, target_label, features_list)

                    if success:
                        session_samples += 1
                        label_counts[target_label] = label_counts.get(target_label, 0) + 1
                        feedback_msg = f"[+] Saved sample #{session_samples} for '{target_label}' (Label Total: {label_counts[target_label]})"
                        feedback_is_error = False
                        feedback_timer = time.time()
                        print(
                            f"[+] Recorded sample #{session_samples}: Label='{target_label}' "
                            f"(Session count for '{target_label}': {label_counts[target_label]}, "
                            f"Features: 63 floats)"
                        )
                    else:
                        feedback_msg = f"[!] Failed to write sample to '{csv_path}'."
                        feedback_is_error = True
                        feedback_timer = time.time()

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        print("\n" + "=" * 65)
        print("  SESSION SUMMARY")
        print("=" * 65)
        print(f"- Total Samples Collected This Session : {session_samples}")
        print(f"- Total Samples in Dataset File        : {total_file_samples + session_samples}")
        if label_counts:
            print("- Per-label breakdown:")
            for lbl, cnt in sorted(label_counts.items()):
                print(f"    * Label '{lbl}': {cnt} samples")
        print(f"- Saved to: {os.path.abspath(csv_path)}")
        print("=" * 65)

        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Resources released cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset collection tool for sign language classification with OpenCV and MediaPipe Hands."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="landmarks.csv",
        help="Path to output CSV file (default: landmarks.csv)",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="A",
        help="Default active label (e.g. A, B, HELLO; default: A)",
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Webcam frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Webcam frame height (default: 480)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum hands to detect (default: 1)",
    )

    args = parser.parse_args()

    run_data_collection(
        csv_path=args.output,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        default_label=args.label,
        max_hands=args.max_hands,
    )


if __name__ == "__main__":
    main()
