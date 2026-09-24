/**
 * main.js - Real-Time Sign Language Interpreter Kiosk Terminal
 * 
 * Powered by:
 * - MediaPipe Tasks Vision (Loaded via CDN)
 * - HTML5 <video> & <canvas>
 * - Fetch API -> Express Backend (http://localhost:3000/predict)
 * - Public Service Kiosk UX: Live Translation, Sentence Builder, Speech Synthesis
 */

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const canvasCtx = canvas.getContext("2d");

// Overlays & Guides
const loaderOverlay = document.getElementById("loader_overlay");
const loaderStatus = document.getElementById("loader_status");
const targetGuide = document.getElementById("target_guide");
const reticleBadge = document.getElementById("reticle_badge");

// Kiosk Telemetry & Badges
const kioskClock = document.getElementById("kiosk_clock");
const backendBadge = document.getElementById("backend_badge");
const backendStatusText = document.getElementById("backend_status_text");
const backendLatencyEl = document.getElementById("backend_latency");
const fpsCounter = document.getElementById("fps_counter");
const ptsCounter = document.getElementById("pts_counter");
const payloadStatus = document.getElementById("payload_status");
const payloadPreview = document.getElementById("payload_preview");

// Prediction Display
const predictionDisplay = document.getElementById("prediction_display");
const confidencePercent = document.getElementById("confidence_percent");
const confidenceBar = document.getElementById("confidence_bar");

// Transcript Elements
const transcriptContent = document.getElementById("transcript_content");
const btnSpeakTranscript = document.getElementById("btn_speak_transcript");
const btnSpace = document.getElementById("btn_space");
const btnBackspace = document.getElementById("btn_backspace");
const btnCopyTranscript = document.getElementById("btn_copy_transcript");
const copyBtnText = document.getElementById("copy_btn_text");
const btnClearTranscript = document.getElementById("btn_clear_transcript");

// Controls
const btnToggleCam = document.getElementById("btn_toggle_cam");
const camBtnText = document.getElementById("cam_btn_text");
const btnToggleMirror = document.getElementById("btn_toggle_mirror");
const btnToggleLandmarks = document.getElementById("btn_toggle_landmarks");

// =============================================================================
// CONFIGURATION & STATE
// =============================================================================
const BACKEND_URL = "http://localhost:3000/predict";
const MEDIAPIPE_CDN_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let handLandmarker = null;
let drawingUtils = null;
let HandLandmarkerClass = null;

let webcamRunning = false;
let showLandmarks = true;
let isMirrored = true;

let lastVideoTime = -1;
let lastPredictTime = 0;
const PREDICT_INTERVAL_MS = 100; // Throttle backend fetch requests (~10 Hz)
let isRequestInFlight = false;

// FPS tracking
let frameCount = 0;
let lastFpsUpdateTime = performance.now();
let currentFps = 0;

// Transcript & Debounce State
let transcriptText = "";
let lastCommittedGesture = "";
let gestureHoldCount = 0;
const GESTURE_COMMIT_THRESHOLD = 8; // Must detect same gesture for 8 consecutive frames to auto-append

// Backend health state
let backendOnline = false;

// =============================================================================
// 1. KIOSK CLOCK
// =============================================================================
function startKioskClock() {
  const updateClock = () => {
    const now = new Date();
    kioskClock.textContent = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };
  updateClock();
  setInterval(updateClock, 1000);
}

// =============================================================================
// 2. LOAD MEDIAPIPE TASKS VISION VIA CDN
// =============================================================================
async function initMediaPipeVision() {
  loaderStatus.textContent = "Loading @mediapipe/tasks-vision from CDN...";

  try {
    // Check if pre-loaded in window or dynamic import via CDN
    let visionModule = window.tasksVision;
    if (!visionModule) {
      visionModule = await import(/* @vite-ignore */ MEDIAPIPE_CDN_URL);
      window.tasksVision = visionModule;
    }

    const { HandLandmarker, FilesetResolver, DrawingUtils } = visionModule;
    HandLandmarkerClass = HandLandmarker;
    drawingUtils = new DrawingUtils(canvasCtx);

    loaderStatus.textContent = "Loading AI Hand Landmark Model...";

    // Initialize WebAssembly fileset
    const vision = await FilesetResolver.forVisionTasks(WASM_CDN_URL);

    // Create HandLandmarker with GPU delegate and CPU fallback
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuError) {
      console.warn("GPU delegate unavailable, falling back to CPU:", gpuError);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
    }

    console.log("[*] MediaPipe HandLandmarker successfully loaded from CDN.");
    loaderStatus.textContent = "Requesting Webcam Access...";
    return true;
  } catch (error) {
    console.error("[!] Failed to initialize MediaPipe Tasks Vision:", error);
    loaderStatus.textContent = "Error loading MediaPipe. Check network connection.";
    return false;
  }
}

// =============================================================================
// 3. WEBCAM STREAM SETUP
// =============================================================================
async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera API not supported by this browser.");
    return false;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadeddata = () => {
        video.play();
        webcamRunning = true;

        // Sync canvas resolution with video stream
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        loaderOverlay.classList.add("hidden");
        btnToggleCam.classList.add("active");
        camBtnText.textContent = "Camera Active";

        console.log(`[*] Webcam active (${canvas.width}x${canvas.height})`);
        resolve(true);
      };
    });
  } catch (err) {
    console.error("[!] Camera access denied or failed:", err);
    loaderStatus.textContent = "Camera access denied. Please grant webcam permissions.";
    return false;
  }
}

function stopWebcam() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    video.srcObject = null;
  }
  webcamRunning = false;
  btnToggleCam.classList.remove("active");
  camBtnText.textContent = "Camera Offline";
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  reticleBadge.textContent = "Camera Offline";
  reticleBadge.className = "reticle-badge searching";
  targetGuide.classList.remove("active");
  fpsCounter.textContent = "--";
  ptsCounter.textContent = "0/21";
}

// =============================================================================
// 4. REAL-TIME LANDMARK PROCESSING LOOP
// =============================================================================
function processLiveVideoFeed() {
  if (!webcamRunning || !handLandmarker) {
    requestAnimationFrame(processLiveVideoFeed);
    return;
  }

  const now = performance.now();

  // Calculate rendering FPS
  frameCount++;
  if (now - lastFpsUpdateTime >= 1000) {
    currentFps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
    fpsCounter.textContent = `${currentFps}`;
    frameCount = 0;
    lastFpsUpdateTime = now;
  }

  // Detect landmarks when new video frame is ready
  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;

    // Run inference for current video timestamp
    const results = handLandmarker.detectForVideo(video, now);

    // Clear canvas overlay
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const handsDetected = results.landmarks && results.landmarks.length > 0;

    if (handsDetected) {
      const primaryHand = results.landmarks[0]; // 21 landmarks

      // Update UI badges
      reticleBadge.textContent = "Hand Detected";
      reticleBadge.className = "reticle-badge detected";
      targetGuide.classList.add("active");
      ptsCounter.textContent = `${primaryHand.length}/21`;

      // Draw skeleton on canvas
      if (showLandmarks && drawingUtils && HandLandmarkerClass) {
        drawingUtils.drawConnectors(primaryHand, HandLandmarkerClass.HAND_CONNECTIONS, {
          color: "#06b6d4", // Cyan connector lines
          lineWidth: 3,
        });
        drawingUtils.drawLandmarks(primaryHand, {
          color: "#10b981", // Emerald keypoints
          fillColor: "#ffffff",
          lineWidth: 1.5,
          radius: 4,
        });
      }

      // Step 4: Flatten the 21 landmarks into a single array of 63 coordinates (x, y, z)
      const flattened63 = extractAndFlattenLandmarks(primaryHand);

      // Update live payload preview monitor
      updatePayloadMonitor(flattened63);

      // Step 5: Send this lightweight array of 63 numbers to local Express backend
      if (now - lastPredictTime >= PREDICT_INTERVAL_MS && !isRequestInFlight) {
        lastPredictTime = now;
        sendLandmarksToBackend(flattened63);
      }
    } else {
      // No hand in view
      reticleBadge.textContent = "Awaiting Hand";
      reticleBadge.className = "reticle-badge searching";
      targetGuide.classList.remove("active");
      ptsCounter.textContent = "0/21";
      payloadStatus.textContent = "Idle";
      payloadPreview.textContent = "[ awaiting hand detection... ]";

      // Reset hold count
      gestureHoldCount = 0;
    }
  }

  requestAnimationFrame(processLiveVideoFeed);
}

// =============================================================================
// 5. EXTRACT & FLATTEN 21 3D LANDMARKS (63 COORDINATES)
// =============================================================================
function extractAndFlattenLandmarks(landmarks) {
  // landmarks is an array of 21 objects: [{ x, y, z }, ...]
  const flattened = new Array(63);
  let idx = 0;

  for (let i = 0; i < landmarks.length && i < 21; i++) {
    const pt = landmarks[i];
    flattened[idx++] = Number(pt.x.toFixed(6));
    flattened[idx++] = Number(pt.y.toFixed(6));
    flattened[idx++] = Number(pt.z.toFixed(6));
  }

  return flattened;
}

function updatePayloadMonitor(coords63) {
  payloadStatus.textContent = "63 floats ready";
  const previewSample = coords63
    .slice(0, 9)
    .map((v) => (v >= 0 ? ` ${v.toFixed(3)}` : v.toFixed(3)))
    .join(", ");
  payloadPreview.textContent = `[ ${previewSample}, ... +54 more ]`;
}

// =============================================================================
// 6. BACKEND COMMUNICATION (FETCH API -> localhost:3000/predict)
// =============================================================================
async function sendLandmarksToBackend(flattened63) {
  if (flattened63.length !== 63) {
    console.warn("Payload does not contain exactly 63 coordinates:", flattened63.length);
    return;
  }

  isRequestInFlight = true;
  payloadStatus.textContent = "Transmitting...";
  const startTime = performance.now();

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        landmarks: flattened63,
        features: flattened63, // Fallback key for diverse express controllers
      }),
    });

    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    setBackendOnline(true);

    // Extract predicted gesture & confidence
    const gesture = data.prediction || data.label || data.gesture || data.sign || "—";
    const confidence = typeof data.confidence === "number" ? data.confidence : (data.probability || 0.95);

    handlePredictionSuccess(gesture, confidence);
  } catch (error) {
    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;
    setBackendOnline(false);

    // Fallback simulated prediction mode so kiosk remains demonstrable before Express server is up
    handleBackendStandby(flattened63);
  } finally {
    isRequestInFlight = false;
  }
}

function setBackendOnline(isOnline) {
  if (isOnline !== backendOnline) {
    backendOnline = isOnline;
    if (isOnline) {
      backendBadge.className = "kiosk-status-badge online";
      backendStatusText.textContent = "Connected: localhost:3000";
    } else {
      backendBadge.className = "kiosk-status-badge standby";
      backendStatusText.textContent = "Standby: localhost:3000";
    }
  }
}

function handlePredictionSuccess(gesture, confidence) {
  // Update Hero prediction display
  predictionDisplay.textContent = gesture;
  predictionDisplay.classList.remove("empty");

  // Animate hero text pulse on new recognition
  predictionDisplay.style.transform = "scale(1.08)";
  setTimeout(() => {
    predictionDisplay.style.transform = "scale(1.0)";
  }, 120);

  // Update confidence bar
  const confPercent = Math.min(100, Math.round(confidence * 100));
  confidencePercent.textContent = `${confPercent}%`;
  confidenceBar.style.width = `${confPercent}%`;

  // Transcript sentence buffer with hold-duration debouncing
  if (gesture !== "—") {
    if (gesture === lastCommittedGesture) {
      gestureHoldCount++;
      if (gestureHoldCount === GESTURE_COMMIT_THRESHOLD) {
        appendGestureToTranscript(gesture);
      }
    } else {
      lastCommittedGesture = gesture;
      gestureHoldCount = 1;
    }
  }
}

// Fallback behavior when Express server is in standby
function handleBackendStandby(flattened63) {
  // Simple heuristic preview: indicate coordinates are streaming
  predictionDisplay.textContent = "•";
  predictionDisplay.classList.remove("empty");
  confidencePercent.textContent = "Ready";
  confidenceBar.style.width = "40%";
  payloadStatus.textContent = "Streaming (Server Standby)";
}

// =============================================================================
// 7. TRANSCRIPT & ACCESSIBILITY SENTENCE BUILDER
// =============================================================================
function appendGestureToTranscript(char) {
  if (char.length === 1) {
    transcriptText += char;
  } else {
    transcriptText += (transcriptText.length > 0 && !transcriptText.endsWith(" ") ? " " : "") + char + " ";
  }
  renderTranscript();
}

function renderTranscript() {
  transcriptContent.textContent = transcriptText;
  const transcriptBox = document.getElementById("transcript_box");
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function speakTranscriptText(text) {
  if (!window.speechSynthesis) {
    alert("Speech Synthesis (TTS) is not supported in this browser.");
    return;
  }
  const phrase = text || transcriptText || predictionDisplay.textContent;
  if (!phrase || phrase === "—" || phrase === "•") return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(phrase);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// =============================================================================
// 8. EVENT LISTENERS & INTERACTIVE KIOSK CONTROLS
// =============================================================================
function initKioskEventListeners() {
  // Camera toggle
  btnToggleCam.addEventListener("click", () => {
    if (webcamRunning) {
      stopWebcam();
    } else {
      loaderOverlay.classList.remove("hidden");
      loaderStatus.textContent = "Activating Webcam...";
      startWebcam();
    }
  });

  // Mirror view toggle
  btnToggleMirror.addEventListener("click", () => {
    isMirrored = !isMirrored;
    video.classList.toggle("mirrored", isMirrored);
    canvas.classList.toggle("mirrored", isMirrored);
    btnToggleMirror.classList.toggle("active", isMirrored);
  });

  // Landmarks overlay toggle
  btnToggleLandmarks.addEventListener("click", () => {
    showLandmarks = !showLandmarks;
    btnToggleLandmarks.classList.toggle("active", showLandmarks);
    if (!showLandmarks) {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
  });

  // Transcript Space
  btnSpace.addEventListener("click", () => {
    transcriptText += " ";
    renderTranscript();
  });

  // Transcript Backspace
  btnBackspace.addEventListener("click", () => {
    transcriptText = transcriptText.slice(0, -1);
    renderTranscript();
  });

  // Transcript Clear
  btnClearTranscript.addEventListener("click", () => {
    transcriptText = "";
    lastCommittedGesture = "";
    gestureHoldCount = 0;
    renderTranscript();
  });

  // Transcript Copy to Clipboard
  btnCopyTranscript.addEventListener("click", async () => {
    if (!transcriptText) return;
    try {
      await navigator.clipboard.writeText(transcriptText);
      copyBtnText.textContent = "Copied!";
      setTimeout(() => {
        copyBtnText.textContent = "Copy";
      }, 1500);
    } catch (e) {
      console.error("Clipboard copy failed:", e);
    }
  });

  // Text-to-Speech Speak Aloud
  btnSpeakTranscript.addEventListener("click", () => {
    speakTranscriptText();
  });

  // Keyboard accessibility shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === " ") {
      e.preventDefault();
      transcriptText += " ";
      renderTranscript();
    } else if (e.key === "Backspace") {
      transcriptText = transcriptText.slice(0, -1);
      renderTranscript();
    } else if (e.key === "Enter") {
      speakTranscriptText();
    }
  });
}

// =============================================================================
// 9. APPLICATION BOOTSTRAP
// =============================================================================
async function bootstrap() {
  console.log("[*] Initializing BridgeSign Public Service Terminal...");
  startKioskClock();
  initKioskEventListeners();

  const visionReady = await initMediaPipeVision();
  if (visionReady) {
    const cameraReady = await startWebcam();
    if (cameraReady) {
      processLiveVideoFeed();
    }
  }
}

// Start application when DOM is loaded
window.addEventListener("DOMContentLoaded", bootstrap);
