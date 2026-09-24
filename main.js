/**
 * main.js - Real-Time Dual-Hand (42 Points) Sign Language Interpreter Kiosk Terminal
 * 
 * Powered by:
 * - MediaPipe Tasks Vision (Dual Hand Tracking: numHands=2, 42 landmarks)
 * - HTML5 <video> & <canvas> with custom dual-color skeletal rendering
 * - High-speed Real-Time Gesture Calculator Engine (Geometric & Kinematic)
 * - Public Service Accessibility Chatbox with auto-send, speech synthesis (TTS), & conversation history
 * - Fetch API -> Express Backend (http://localhost:3000/predict) transmitting 126-coordinate payloads
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
const reticleBadge = document.getElementById("reticle_badge");
const hand1Indicator = document.getElementById("hand1_indicator");
const hand2Indicator = document.getElementById("hand2_indicator");

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
const predictionSubtext = document.getElementById("prediction_subtext");
const confidencePercent = document.getElementById("confidence_percent");
const confidenceBar = document.getElementById("confidence_bar");
const confidenceLabel = document.getElementById("confidence_label");
const vocabGrid = document.getElementById("vocab_grid");

// Chatbox Elements
const chatMessages = document.getElementById("chat_messages");
const chatInput = document.getElementById("chat_input");
const btnSendChat = document.getElementById("btn_send_chat");
const btnSpace = document.getElementById("btn_space");
const btnBackspace = document.getElementById("btn_backspace");
const btnSpeakTranscript = document.getElementById("btn_speak_transcript");

const btnToggleTts = document.getElementById("btn_toggle_tts");
const btnToggleAutosend = document.getElementById("btn_toggle_autosend");
const btnExportChat = document.getElementById("btn_export_chat");
const btnClearChat = document.getElementById("btn_clear_chat");

// Staging Banner
const gestureStagingBar = document.getElementById("gesture_staging_bar");
const stagingGestureName = document.getElementById("staging_gesture_name");
const stagingConfBadge = document.getElementById("staging_conf_badge");
const stagingProgressFill = document.getElementById("staging_progress_fill");
const stagingStatusText = document.getElementById("staging_status_text");


// Controls
const btnToggleCam = document.getElementById("btn_toggle_cam");
const camBtnText = document.getElementById("cam_btn_text");
const btnToggleMirror = document.getElementById("btn_toggle_mirror");
const btnToggleLandmarks = document.getElementById("btn_toggle_landmarks");
const btnTogglePip = document.getElementById("btn_toggle_pip");
const pipBtnText = document.getElementById("pip_btn_text");
const viewportPanel = document.querySelector(".viewport-panel");

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

// Chat & Auto-Send Options
let isTtsEnabled = true;
let isAutoSendEnabled = true;
const AUTO_SEND_HOLD_MS = 1100; // Hold gesture for 1.1 seconds to commit to chat

// Gesture Tracking & Staging State
let activeGestureObj = null;
let stagingGestureKey = "";
let stagingStartTime = 0;
let lastCommittedGestureKey = "";
let lastCommitTimestamp = 0;
const COMMIT_COOLDOWN_MS = 1600; // Prevent spamming duplicate commit within 1.6s

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
// 2. LOAD MEDIAPIPE TASKS VISION (DUAL HAND: numHands = 2)
// =============================================================================
async function initMediaPipeVision() {
  loaderStatus.textContent = "Loading @mediapipe/tasks-vision CDN...";

  try {
    let visionModule = window.tasksVision;
    if (!visionModule) {
      visionModule = await import(/* @vite-ignore */ MEDIAPIPE_CDN_URL);
      window.tasksVision = visionModule;
    }

    const { HandLandmarker, FilesetResolver, DrawingUtils } = visionModule;
    HandLandmarkerClass = HandLandmarker;
    drawingUtils = new DrawingUtils(canvasCtx);

    loaderStatus.textContent = "Loading Dual-Hand 42-Point Landmark Model...";

    const vision = await FilesetResolver.forVisionTasks(WASM_CDN_URL);

    // Create HandLandmarker with 2 hands support (42 points total)
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuError) {
      console.warn("[!] GPU delegate failed, falling back to CPU:", gpuError);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    }

    console.log("[*] MediaPipe HandLandmarker configured for 2 Hands (42 points).");
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

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        loaderOverlay.classList.add("hidden");
        btnToggleCam.classList.add("active");
        camBtnText.textContent = "Camera Active";

        console.log(`[*] Webcam online: ${canvas.width}x${canvas.height}`);
        resolve(true);
      };
    });
  } catch (err) {
    console.error("[!] Camera access denied:", err);
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
  fpsCounter.textContent = "--";
  ptsCounter.textContent = "0/42";
  hand1Indicator.className = "hand-chip offline";
  hand2Indicator.className = "hand-chip offline";
}

// =============================================================================
// 4. REAL-TIME GEOMETRIC GESTURE CALCULATION ENGINE (1 & 2 HANDS / 42 PTS)
// =============================================================================
function dist2D(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function getFingerStates(landmarks) {
  // landmarks has 21 points
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];

  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const indexMcp = landmarks[5];

  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const middleMcp = landmarks[9];

  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const ringMcp = landmarks[13];

  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];
  const pinkyMcp = landmarks[17];

  // Extension checked against PIP and distance from wrist
  const indexExt = indexTip.y < indexPip.y && dist2D(indexTip, wrist) > dist2D(indexPip, wrist);
  const middleExt = middleTip.y < middlePip.y && dist2D(middleTip, wrist) > dist2D(middlePip, wrist);
  const ringExt = ringTip.y < ringPip.y && dist2D(ringTip, wrist) > dist2D(ringPip, wrist);
  const pinkyExt = pinkyTip.y < pinkyPip.y && dist2D(pinkyTip, wrist) > dist2D(pinkyPip, wrist);

  // Thumb extended if tip is far from pinky MCP
  const thumbExt = dist2D(thumbTip, pinkyMcp) > dist2D(thumbIp, pinkyMcp) * 1.15;

  return {
    wrist,
    thumbTip,
    thumbMcp,
    thumbIp,
    indexTip,
    indexPip,
    indexMcp,
    middleTip,
    middlePip,
    middleMcp,
    ringTip,
    ringPip,
    ringMcp,
    pinkyTip,
    pinkyPip,
    pinkyMcp,
    thumbExt,
    indexExt,
    middleExt,
    ringExt,
    pinkyExt,
  };
}

// =============================================================================
// 4. BEGINNER SIGN VOCABULARY DICTIONARY (20 WORDS)
// =============================================================================
const BEGINNER_VOCABULARY = [
  {
    key: "HELLO",
    label: "Hello",
    emoji: "👋",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Open palm waving facing forward",
    speech: "Hello! Welcome to citizen services.",
    response: "Hello! Welcome to Counter #04. How can I assist you with your services today?",
  },
  {
    key: "THANK_YOU",
    label: "Thank You",
    emoji: "🙏",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both palms touching together in prayer",
    speech: "Thank you very much.",
    response: "You are very welcome! It is our honor to serve you.",
  },
  {
    key: "YES",
    label: "Yes",
    emoji: "👍",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb pointing upright (thumbs up)",
    speech: "Yes, confirmed.",
    response: "Understood: Confirmed. Proceeding with your application.",
  },
  {
    key: "NO",
    label: "No",
    emoji: "👎",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb pointing downward (thumbs down)",
    speech: "No, decline.",
    response: "Noted: Cancelled. We will not proceed with this action.",
  },
  {
    key: "PLEASE",
    label: "Please",
    emoji: "🤲",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open cupped palms held gently upward",
    speech: "Please assist me.",
    response: "Certainly! We are glad to assist you with every step.",
  },
  {
    key: "HELP",
    label: "Help",
    emoji: "🆘",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Fist resting on flat palm of other hand",
    speech: "I need assistance.",
    response: "Assistance alert acknowledged. A public service officer is attending to you.",
  },
  {
    key: "GOOD",
    label: "Good",
    emoji: "👌",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb and index touching in circle (OK)",
    speech: "Very good.",
    response: "Great! Glad to hear everything is going smoothly.",
  },
  {
    key: "BAD",
    label: "Bad",
    emoji: "⛔",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Hand facing downward with fingers lowered",
    speech: "There is an issue.",
    response: "We apologize for the inconvenience. Let us resolve this for you.",
  },
  {
    key: "LOVE",
    label: "I Love You",
    emoji: "🤟",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Thumb, index, and pinky extended (ASL ILY)",
    speech: "I love you.",
    response: "Much love, warmth, and respect right back to you! 🤟",
  },
  {
    key: "HEART",
    label: "Heart",
    emoji: "❤️",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Thumbs and index fingertips joined in heart shape",
    speech: "Kindness from the heart.",
    response: "Heartfelt kindness received! Wishing you peace and happiness.",
  },
  {
    key: "STOP",
    label: "Stop",
    emoji: "🛑",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both wrists crossed in front forming an X",
    speech: "Stop and pause.",
    response: "Process halted immediately. Take your time to review.",
  },
  {
    key: "SORRY",
    label: "Sorry",
    emoji: "🙇",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "Closed fist held firmly over chest",
    speech: "I am sorry.",
    response: "No worries at all! Everything is completely fine.",
  },
  {
    key: "FRIEND",
    label: "Friend",
    emoji: "🤝",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both index fingers hooked or wrists linked",
    speech: "We are friends.",
    response: "Welcome, dear friend! You are always supported and valued here.",
  },
  {
    key: "MORE",
    label: "More",
    emoji: "🤏",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Fingertips of both hands touching together",
    speech: "I would like more.",
    response: "Displaying additional options for your request.",
  },
  {
    key: "WATER",
    label: "Water",
    emoji: "💧",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "W sign: Index, middle, and ring extended upright",
    speech: "Drinking water.",
    response: "Drinking water dispenser is located next to Counter #04.",
  },
  {
    key: "FOOD",
    label: "Food",
    emoji: "🍽️",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open palms held side by side facing upward",
    speech: "Food and dining.",
    response: "Community dining and cafeteria facilities are on Level 1.",
  },
  {
    key: "WELCOME",
    label: "Welcome",
    emoji: "👐",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both open palms spread wide welcoming forward",
    speech: "Welcome!",
    response: "A very warm welcome to Counter #04! How can I assist you?",
  },
  {
    key: "PEACE",
    label: "Peace",
    emoji: "✌️",
    hands: "1H",
    handsCount: 1,
    pointsCount: 21,
    tip: "V sign: Index and middle fingers extended",
    speech: "Peace to you.",
    response: "Peace and harmony to you and your community!",
  },
  {
    key: "TOGETHER",
    label: "Together",
    emoji: "👥",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Both fists touching side by side",
    speech: "We work together.",
    response: "Together we accomplish more! United in service.",
  },
  {
    key: "DONE",
    label: "Done",
    emoji: "✅",
    hands: "2H",
    handsCount: 2,
    pointsCount: 42,
    tip: "Dual thumbs up / palms spread outward",
    speech: "Finished and done.",
    response: "Application marked as completed! Have an excellent day ahead.",
  },
];

function renderVocabDeck() {
  if (!vocabGrid) return;
  vocabGrid.innerHTML = "";

  BEGINNER_VOCABULARY.forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "vocab-chip";
    chip.setAttribute("data-key", item.key);
    chip.title = `${item.label} (${item.hands}) - ${item.tip}`;

    chip.innerHTML = `
      <span class="vocab-chip-emoji">${item.emoji}</span>
      <span class="vocab-chip-label">${item.label}</span>
      <span class="vocab-chip-tag">${item.hands}</span>
    `;

    chip.addEventListener("click", () => {
      const match = buildVocabMatch(item.key, 0.99);
      if (match) {
        highlightVocabChip(item.key);
        handleGestureCalculation(match, match.handsCount, performance.now());
        if (isTtsEnabled) {
          speakText(`${item.label}. ${item.tip}`);
        }
      }
    });

    vocabGrid.appendChild(chip);
  });
}

function highlightVocabChip(vocabKey) {
  if (!vocabGrid) return;
  const chips = vocabGrid.querySelectorAll(".vocab-chip");
  chips.forEach((c) => {
    if (c.getAttribute("data-key") === vocabKey) {
      c.classList.add("active");
      c.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      c.classList.remove("active");
    }
  });
}

function buildVocabMatch(key, confidence) {
  const item = BEGINNER_VOCABULARY.find((v) => v.key === key);
  if (!item) return null;
  return {
    ...item,
    confidence,
    fullText: `${item.emoji} ${item.label} (${item.tip})`,
  };
}

function calculateRealTimeGesture(handsLandmarks) {
  if (!handsLandmarks || handsLandmarks.length === 0) {
    return null;
  }

  const handsCount = handsLandmarks.length;
  const h1 = getFingerStates(handsLandmarks[0]);

  // ---------------------------------------------------------------------------
  // A. TWO-HAND BEGINNER VOCABULARY (42 Points)
  // ---------------------------------------------------------------------------
  if (handsCount >= 2) {
    const h2 = getFingerStates(handsLandmarks[1]);

    const wristDist = dist2D(h1.wrist, h2.wrist);
    const indexTipsDist = dist2D(h1.indexTip, h2.indexTip);
    const thumbTipsDist = dist2D(h1.thumbTip, h2.thumbTip);
    const middleTipsDist = dist2D(h1.middleTip, h2.middleTip);
    const pinkyTipsDist = dist2D(h1.pinkyTip, h2.pinkyTip);

    const allH1Up = h1.indexExt && h1.middleExt && h1.ringExt;
    const allH2Up = h2.indexExt && h2.middleExt && h2.ringExt;

    // 1. HEART (42 pts)
    if (
      thumbTipsDist < 0.10 &&
      indexTipsDist < 0.11 &&
      !h1.pinkyExt &&
      !h2.pinkyExt &&
      wristDist < 0.34
    ) {
      return buildVocabMatch("HEART", 0.98);
    }

    // 2. STOP (42 pts - Crossed Wrists X)
    if (wristDist < 0.13 && indexTipsDist > 0.22) {
      return buildVocabMatch("STOP", 0.96);
    }

    // 3. THANK YOU / NAMASTE (42 pts - Both palms touching upright prayer)
    if (
      allH1Up &&
      allH2Up &&
      wristDist < 0.28 &&
      indexTipsDist < 0.14 &&
      middleTipsDist < 0.14 &&
      thumbTipsDist < 0.16
    ) {
      return buildVocabMatch("THANK_YOU", 0.98);
    }

    // 4. FRIEND (42 pts - Index fingers hooked / close, other fingers curled)
    if (
      indexTipsDist < 0.10 &&
      h1.indexExt &&
      h2.indexExt &&
      !h1.ringExt &&
      !h2.ringExt &&
      !h1.pinkyExt &&
      !h2.pinkyExt
    ) {
      return buildVocabMatch("FRIEND", 0.95);
    }

    // 5. HELP (42 pts - One fist resting on flat palm of other hand)
    const h1FlatH2Fist = allH1Up && !h2.indexExt && !h2.middleExt;
    const h2FlatH1Fist = allH2Up && !h1.indexExt && !h1.middleExt;
    if ((h1FlatH2Fist || h2FlatH1Fist) && wristDist < 0.28) {
      return buildVocabMatch("HELP", 0.95);
    }

    // 6. MORE (42 pts - Both hands pinched, tips touching)
    if (
      indexTipsDist < 0.09 &&
      thumbTipsDist < 0.09 &&
      !h1.ringExt &&
      !h2.ringExt &&
      wristDist < 0.25
    ) {
      return buildVocabMatch("MORE", 0.94);
    }

    // 7. TOGETHER (42 pts - Both fists touching)
    if (
      wristDist < 0.25 &&
      !h1.indexExt &&
      !h1.middleExt &&
      !h2.indexExt &&
      !h2.middleExt
    ) {
      return buildVocabMatch("TOGETHER", 0.94);
    }

    // 8. DONE (42 pts - Dual thumbs up)
    const h1ThumbUp = h1.thumbExt && !h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt && h1.thumbTip.y < h1.thumbMcp.y;
    const h2ThumbUp = h2.thumbExt && !h2.indexExt && !h2.middleExt && !h2.ringExt && !h2.pinkyExt && h2.thumbTip.y < h2.thumbMcp.y;
    if (h1ThumbUp && h2ThumbUp) {
      return buildVocabMatch("DONE", 0.98);
    }

    // 9. FOOD (42 pts - Both open palms held upward side-by-side like plate/book)
    if (
      pinkyTipsDist < 0.16 &&
      wristDist < 0.26 &&
      allH1Up &&
      allH2Up &&
      h1.indexTip.y > h1.wrist.y - 0.2
    ) {
      return buildVocabMatch("FOOD", 0.94);
    }

    // 10. PLEASE (42 pts - Both open palms cupped together gently)
    if (allH1Up && allH2Up && wristDist < 0.25 && indexTipsDist < 0.20) {
      return buildVocabMatch("PLEASE", 0.93);
    }

    // 11. WELCOME (42 pts - Both open palms spread wide welcoming)
    if (allH1Up && allH2Up && h1.pinkyExt && h2.pinkyExt && wristDist > 0.32) {
      return buildVocabMatch("WELCOME", 0.95);
    }
  }

  // ---------------------------------------------------------------------------
  // B. SINGLE-HAND BEGINNER VOCABULARY (21 Points)
  // ---------------------------------------------------------------------------
  // 12. I LOVE YOU (ASL ILY - Thumb + Index + Pinky)
  if (h1.thumbExt && h1.indexExt && !h1.middleExt && !h1.ringExt && h1.pinkyExt) {
    return buildVocabMatch("LOVE", 0.97);
  }

  // 13. GOOD (OK Sign - Thumb and Index circle)
  const okDist = dist2D(h1.thumbTip, h1.indexTip);
  if (okDist < 0.06 && h1.middleExt && h1.ringExt && h1.pinkyExt) {
    return buildVocabMatch("GOOD", 0.96);
  }

  // 14. WATER (W sign - Index + Middle + Ring upright)
  if (h1.indexExt && h1.middleExt && h1.ringExt && !h1.pinkyExt) {
    return buildVocabMatch("WATER", 0.95);
  }

  // 15. PEACE (V sign - Index + Middle upright)
  if (h1.indexExt && h1.middleExt && !h1.ringExt && !h1.pinkyExt) {
    return buildVocabMatch("PEACE", 0.96);
  }

  // 16. YES (Thumb Up)
  if (
    h1.thumbExt &&
    !h1.indexExt &&
    !h1.middleExt &&
    !h1.ringExt &&
    !h1.pinkyExt &&
    h1.thumbTip.y < h1.thumbMcp.y
  ) {
    return buildVocabMatch("YES", 0.97);
  }

  // 17. NO (Thumb Down)
  if (
    h1.thumbExt &&
    !h1.indexExt &&
    !h1.middleExt &&
    !h1.ringExt &&
    !h1.pinkyExt &&
    h1.thumbTip.y > h1.thumbMcp.y + 0.05
  ) {
    return buildVocabMatch("NO", 0.95);
  }

  // 18. HELLO (Open Palm Wave)
  if (h1.thumbExt && h1.indexExt && h1.middleExt && h1.ringExt && h1.pinkyExt) {
    return buildVocabMatch("HELLO", 0.94);
  }

  // 19. BAD (Hand turned downward)
  if (
    h1.indexTip.y > h1.wrist.y &&
    h1.middleTip.y > h1.wrist.y &&
    !h1.thumbExt
  ) {
    return buildVocabMatch("BAD", 0.92);
  }

  // 20. SORRY (Closed Fist over chest)
  if (!h1.indexExt && !h1.middleExt && !h1.ringExt && !h1.pinkyExt) {
    return buildVocabMatch("SORRY", 0.92);
  }

  return null;
}

// =============================================================================
// 5. EXTRACT & FLATTEN DUAL-HAND LANDMARKS (126 FLOATS = 42 POINTS × 3)
// =============================================================================
function extractDualHandCoordinates(handsLandmarks) {
  // Returns 126 floats (42 points total)
  const coords = new Array(126).fill(0.0);

  if (!handsLandmarks || handsLandmarks.length === 0) {
    return coords;
  }

  // Hand 1 (indices 0..62)
  const h1 = handsLandmarks[0];
  for (let i = 0; i < h1.length && i < 21; i++) {
    const pt = h1[i];
    coords[i * 3] = Number(pt.x.toFixed(6));
    coords[i * 3 + 1] = Number(pt.y.toFixed(6));
    coords[i * 3 + 2] = Number(pt.z.toFixed(6));
  }

  // Hand 2 (indices 63..125)
  if (handsLandmarks.length > 1) {
    const h2 = handsLandmarks[1];
    for (let i = 0; i < h2.length && i < 21; i++) {
      const pt = h2[i];
      coords[63 + i * 3] = Number(pt.x.toFixed(6));
      coords[63 + i * 3 + 1] = Number(pt.y.toFixed(6));
      coords[63 + i * 3 + 2] = Number(pt.z.toFixed(6));
    }
  }

  return coords;
}

function updatePayloadMonitor(coords126, handsCount) {
  const pointsCount = handsCount * 21;
  payloadStatus.textContent = `${pointsCount}/42 pts (${coords126.length} floats)`;

  const sample = coords126
    .slice(0, 8)
    .map((v) => (v >= 0 ? ` ${v.toFixed(3)}` : v.toFixed(3)))
    .join(", ");
  payloadPreview.textContent = `[ ${sample}, ... +118 more (Hands: ${handsCount}) ]`;
}

// =============================================================================
// 6. SKELETAL RENDERING WITH DISTINCT DUAL-HAND VISUALS
// =============================================================================
function renderDualHandSkeletons(handsLandmarks) {
  if (!showLandmarks || !drawingUtils || !HandLandmarkerClass || !handsLandmarks) return;

  const w = canvas.width;
  const h = canvas.height;

  handsLandmarks.forEach((hand, idx) => {
    // Hand 1: Warm Tangerine (#F26522) & White (#FFFFFF); Hand 2: Light Gray (#EBEBEB) & Orange (#F26522)
    const isHand1 = idx === 0;
    const connectorColor = isHand1 ? "#F26522" : "#EBEBEB";
    const landmarkColor = isHand1 ? "#FFFFFF" : "#F26522";

    drawingUtils.drawConnectors(hand, HandLandmarkerClass.HAND_CONNECTIONS, {
      color: connectorColor,
      lineWidth: 3.5,
    });
    drawingUtils.drawLandmarks(hand, {
      color: landmarkColor,
      fillColor: "#FFFFFF",
      lineWidth: 1.5,
      radius: 4,
    });

    // Draw Wrist Floating Badge Label directly on canvas
    if (hand.length > 0) {
      const wrist = hand[0];
      const wx = wrist.x * w;
      const wy = wrist.y * h;

      canvasCtx.save();
      canvasCtx.font = "bold 11px 'JetBrains Mono', monospace";
      const labelText = isHand1 ? "HAND 1: TANGERINE (21 pts)" : "HAND 2: WHITE (21 pts)";
      const metrics = canvasCtx.measureText(labelText);
      const padding = 5;

      canvasCtx.fillStyle = "rgba(26, 26, 26, 0.9)";
      canvasCtx.strokeStyle = connectorColor;
      canvasCtx.lineWidth = 1;
      canvasCtx.beginPath();
      canvasCtx.roundRect(wx - 6, wy - 22, metrics.width + padding * 2, 18, 4);
      canvasCtx.fill();
      canvasCtx.stroke();

      canvasCtx.fillStyle = landmarkColor;
      canvasCtx.fillText(labelText, wx - 6 + padding, wy - 9);
      canvasCtx.restore();
    }
  });
}

// =============================================================================
// 7. REAL-TIME LANDMARK & GESTURE PROCESSING LOOP
// =============================================================================
function processLiveVideoFeed() {
  if (!webcamRunning || !handLandmarker) {
    requestAnimationFrame(processLiveVideoFeed);
    return;
  }

  const now = performance.now();

  // Rendering FPS calculation
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

    const results = handLandmarker.detectForVideo(video, now);

    // Clear canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = results.landmarks || [];
    const handsCount = hands.length;
    const pointsCount = Math.min(handsCount, 2) * 21;

    // Update Telemetry indicators
    ptsCounter.textContent = `${pointsCount}/42`;

    if (handsCount >= 2) {
      reticleBadge.textContent = "2 Hands Detected (42 pts)";
      reticleBadge.className = "reticle-badge dual";
      hand1Indicator.className = "hand-chip active-cyan";
      hand2Indicator.className = "hand-chip active-amber";
    } else if (handsCount === 1) {
      reticleBadge.textContent = "1 Hand Detected (21 pts)";
      reticleBadge.className = "reticle-badge detected";
      hand1Indicator.className = "hand-chip active-cyan";
      hand2Indicator.className = "hand-chip offline";
    } else {
      reticleBadge.textContent = "Awaiting Hands";
      reticleBadge.className = "reticle-badge searching";
      hand1Indicator.className = "hand-chip offline";
      hand2Indicator.className = "hand-chip offline";
    }

    if (handsCount > 0) {
      // 1. Draw dual hand skeleton
      renderDualHandSkeletons(hands);

      // 2. Flatten coordinates into 126 floats (42 points x 3)
      const coords126 = extractDualHandCoordinates(hands);
      updatePayloadMonitor(coords126, handsCount);

      // 3. Real-Time Gesture Calculation
      const detectedGesture = calculateRealTimeGesture(hands);
      handleGestureCalculation(detectedGesture, handsCount, now);

      // 4. Concurrently transmit coordinates to local Express backend for ML telemetry
      if (now - lastPredictTime >= PREDICT_INTERVAL_MS && !isRequestInFlight) {
        lastPredictTime = now;
        sendLandmarksToBackend(coords126);
      }
    } else {
      // No hands present
      resetGestureStaging();
      predictionDisplay.textContent = "—";
      predictionDisplay.classList.add("empty");
      predictionSubtext.textContent = "Show one or both hands (42 points)";
      confidencePercent.textContent = "0.0%";
      confidenceBar.style.width = "0%";
      payloadStatus.textContent = "Idle (0/42 pts)";
      payloadPreview.textContent = "[ awaiting hand detection (0/42 points)... ]";
    }
  }

  requestAnimationFrame(processLiveVideoFeed);
}

// =============================================================================
// 8. REAL-TIME GESTURE CALCULATION & AUTO-COMMIT LOGIC
// =============================================================================
function handleGestureCalculation(gestureObj, handsCount, now) {
  if (!gestureObj) {
    resetGestureStaging();
    predictionDisplay.textContent = "…";
    predictionDisplay.classList.remove("empty");
    predictionSubtext.textContent = handsCount >= 2 ? "Calculating dual-hand vocabulary..." : "Calculating single-hand vocabulary...";
    return;
  }

  activeGestureObj = gestureObj;
  highlightVocabChip(gestureObj.key);

  // Update Hero Card
  predictionDisplay.textContent = `${gestureObj.emoji} ${gestureObj.label}`;
  predictionDisplay.classList.remove("empty");
  predictionSubtext.textContent = `${gestureObj.label} • ${gestureObj.tip || gestureObj.fullText} (${gestureObj.pointsCount} pts)`;

  const confPercent = Math.round(gestureObj.confidence * 100);
  confidencePercent.textContent = `${confPercent}%`;
  confidenceBar.style.width = `${confPercent}%`;
  confidenceLabel.textContent = gestureObj.handsCount === 2 ? "Dual-Hand Accuracy (42 pts)" : "Single-Hand Accuracy (21 pts)";

  // Real-Time Staging & Auto-Send Buffer
  gestureStagingBar.classList.add("active");
  stagingGestureName.textContent = `${gestureObj.emoji} ${gestureObj.label}`;
  stagingConfBadge.textContent = `${confPercent}%`;

  if (stagingGestureKey !== gestureObj.key) {
    // New vocabulary word recognized, start hold timer
    stagingGestureKey = gestureObj.key;
    stagingStartTime = now;
    stagingProgressFill.style.width = "0%";
    stagingStatusText.textContent = isAutoSendEnabled ? "Hold steady to send..." : "Ready to Send";
  } else {
    // Same vocabulary word held
    const elapsed = now - stagingStartTime;
    const progress = Math.min(100, Math.round((elapsed / AUTO_SEND_HOLD_MS) * 100));
    stagingProgressFill.style.width = `${progress}%`;

    const remainingSec = Math.max(0, (AUTO_SEND_HOLD_MS - elapsed) / 1000).toFixed(1);
    stagingStatusText.textContent = isAutoSendEnabled
      ? `Hold: ${remainingSec}s to send`
      : "Press Send 🚀 or Enter";

    // Auto-Send commit trigger
    if (isAutoSendEnabled && elapsed >= AUTO_SEND_HOLD_MS) {
      if (now - lastCommitTimestamp > COMMIT_COOLDOWN_MS || lastCommittedGestureKey !== gestureObj.key) {
        commitGestureToChat(gestureObj);
        lastCommittedGestureKey = gestureObj.key;
        lastCommitTimestamp = now;
        stagingStartTime = now; // reset
        stagingProgressFill.style.width = "0%";
      }
    }
  }
}

function resetGestureStaging() {
  stagingGestureKey = "";
  stagingStartTime = 0;
  stagingProgressFill.style.width = "0%";
  stagingStatusText.textContent = "Hold to commit";
  gestureStagingBar.classList.remove("active");
  stagingGestureName.textContent = "Waiting for sign...";
  stagingConfBadge.textContent = "--%";
  highlightVocabChip(null);
}

// =============================================================================
// 9. CHATBOX MESSAGES & INTERACTION ENGINE
// =============================================================================
function commitGestureToChat(gestureObj) {
  const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
  const pointsBadge = gestureObj.handsCount === 2 ? "2 Hands • 42 Points" : "1 Hand • 21 Points";

  // 1. Post Signer vocabulary message to Chatbox
  addChatMessage({
    sender: "You (Signer)",
    text: `${gestureObj.emoji} ${gestureObj.label}`,
    meta: `Vocabulary: ${gestureObj.label} • ${pointsBadge} • ${timestamp} • ${Math.round(gestureObj.confidence * 100)}% Match`,
    isUser: true,
  });

  // Speak aloud if TTS enabled
  if (isTtsEnabled) {
    speakText(gestureObj.speech || gestureObj.label);
  }

  // 2. Post AI Assistant response after a brief natural pause (400ms)
  if (gestureObj.response) {
    setTimeout(() => {
      const respTime = new Date().toLocaleTimeString("en-US", { hour12: false });
      addChatMessage({
        sender: "🤖 Counter #04 Interpreter",
        text: gestureObj.response,
        meta: `Civic Terminal Assistant • ${respTime}`,
        isUser: false,
      });

      if (isTtsEnabled) {
        setTimeout(() => speakText(gestureObj.response), 300);
      }
    }, 450);
  }
}

function addChatMessage({ sender, text, meta, isUser }) {
  const msgEl = document.createElement("div");
  msgEl.className = `chat-msg ${isUser ? "user" : "assistant"}`;

  msgEl.innerHTML = `
    <div class="chat-msg-header">
      <span class="sender-name">${sender}</span>
      <span class="msg-meta">${meta}</span>
    </div>
    <div class="chat-msg-body">
      ${text}
      <div class="chat-bubble-actions">
        <button class="chat-mini-btn btn-speak-bubble" title="Read message aloud">🔊 Speak</button>
        <button class="chat-mini-btn btn-copy-bubble" title="Copy message text">📋 Copy</button>
      </div>
    </div>
  `;

  // Attach bubble actions
  const speakBtn = msgEl.querySelector(".btn-speak-bubble");
  speakBtn.addEventListener("click", () => speakText(text));

  const copyBtn = msgEl.querySelector(".btn-copy-bubble");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "📋 Copy"), 1200);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  });

  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function speakText(text) {
  if (!window.speechSynthesis) return;

  // Clean emoji characters from speech readout for natural voice output
  const cleanText = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "").trim();
  if (!cleanText) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// =============================================================================
// 10. BACKEND COMMUNICATION (FETCH API -> localhost:3000/predict)
// =============================================================================
async function sendLandmarksToBackend(coords126) {
  isRequestInFlight = true;
  payloadStatus.textContent = "Transmitting (126 floats)...";
  const startTime = performance.now();

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        landmarks: coords126,
        features: coords126,
      }),
    });

    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    setBackendOnline(true);
  } catch (error) {
    const elapsed = Math.round(performance.now() - startTime);
    backendLatencyEl.textContent = `${elapsed} ms`;
    setBackendOnline(false);
  } finally {
    isRequestInFlight = false;
  }
}

function setBackendOnline(isOnline) {
  if (isOnline !== backendOnline) {
    backendOnline = isOnline;
    if (isOnline) {
      backendBadge.className = "kiosk-status-badge online";
      backendStatusText.textContent = "Connected: localhost:3000 (126-Pt API)";
    } else {
      backendBadge.className = "kiosk-status-badge standby";
      backendStatusText.textContent = "Standby: localhost:3000";
    }
  }
}

// =============================================================================
// 11. EVENT LISTENERS & CHATBOX CONTROLS
// =============================================================================
function initKioskEventListeners() {
  // 1. Camera Toggle
  btnToggleCam.addEventListener("click", () => {
    if (webcamRunning) {
      stopWebcam();
    } else {
      loaderOverlay.classList.remove("hidden");
      loaderStatus.textContent = "Activating Webcam...";
      startWebcam();
    }
  });

  // 2. Mirror View Toggle
  btnToggleMirror.addEventListener("click", () => {
    isMirrored = !isMirrored;
    video.classList.toggle("mirrored", isMirrored);
    canvas.classList.toggle("mirrored", isMirrored);
    btnToggleMirror.classList.toggle("active", isMirrored);
  });

  // 3. Landmarks Overlay Toggle
  btnToggleLandmarks.addEventListener("click", () => {
    showLandmarks = !showLandmarks;
    btnToggleLandmarks.classList.toggle("active", showLandmarks);
    if (!showLandmarks) {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
  });

  // 3b. Float Picture-in-Picture Toggle
  if (btnTogglePip && viewportPanel) {
    btnTogglePip.addEventListener("click", () => {
      const isPip = viewportPanel.classList.toggle("pip-mode");
      btnTogglePip.classList.toggle("active", isPip);
      if (pipBtnText) {
        pipBtnText.textContent = isPip ? "📌 Dock View" : "Float PiP";
      }
    });
  }

  // 4. TTS Readout Toggle
  btnToggleTts.addEventListener("click", () => {
    isTtsEnabled = !isTtsEnabled;
    btnToggleTts.classList.toggle("active", isTtsEnabled);
    btnToggleTts.querySelector("span").textContent = isTtsEnabled ? "🔊 Speech ON" : "🔇 Speech OFF";
  });

  // 5. Auto-Send Toggle
  btnToggleAutosend.addEventListener("click", () => {
    isAutoSendEnabled = !isAutoSendEnabled;
    btnToggleAutosend.classList.toggle("active", isAutoSendEnabled);
    btnToggleAutosend.querySelector("span").textContent = isAutoSendEnabled ? "⚡ Auto-Send ON" : "⏸️ Manual Send";
  });

  // 6. Clear Chat
  btnClearChat.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="chat-msg assistant">
        <div class="chat-msg-header">
          <span class="sender-name">🤖 Counter #04 Interpreter</span>
          <span class="msg-meta">Chat Cleared</span>
        </div>
        <div class="chat-msg-body">
          Chat cleared. Point one or both hands (42 points) toward the camera to begin a new real-time translation session.
        </div>
      </div>
    `;
    resetGestureStaging();
  });

  // 7. Export Chat Transcript
  btnExportChat.addEventListener("click", () => {
    const messages = chatMessages.querySelectorAll(".chat-msg");
    if (!messages || messages.length === 0) return;

    let transcript = "=== SIGN LANGUAGE BRIDGE - ACCESSIBILITY TRANSCRIPT ===\n";
    transcript += `Generated: ${new Date().toLocaleString()}\n`;
    transcript += "Counter #04 • Public Hub (42 Points Dual Hand Tracking)\n";
    transcript += "=".repeat(60) + "\n\n";

    messages.forEach((m) => {
      const sender = m.querySelector(".sender-name")?.textContent || "Speaker";
      const meta = m.querySelector(".msg-meta")?.textContent || "";
      const body = m.querySelector(".chat-msg-body")?.childNodes[0]?.textContent?.trim() || "";
      transcript += `[${meta}] ${sender}:\n${body}\n\n`;
    });

    const blob = new Blob([transcript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sign_language_chat_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // 8. Send Button
  btnSendChat.addEventListener("click", () => {
    sendCurrentInputOrActiveGesture();
  });

  // 9. Input Keyboard Enter
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendCurrentInputOrActiveGesture();
    }
  });

  // 10. Space button
  btnSpace.addEventListener("click", () => {
    chatInput.value += " ";
    chatInput.focus();
  });

  // 11. Backspace button
  btnBackspace.addEventListener("click", () => {
    chatInput.value = chatInput.value.slice(0, -1);
    chatInput.focus();
  });

  // 12. Speak latest message
  btnSpeakTranscript.addEventListener("click", () => {
    const lastMsg = chatMessages.querySelector(".chat-msg:last-child .chat-msg-body");
    if (lastMsg) {
      speakText(lastMsg.textContent);
    }
  });


}

function sendCurrentInputOrActiveGesture() {
  const text = chatInput.value.trim();

  if (text) {
    const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
    addChatMessage({
      sender: "You (Signer)",
      text: text,
      meta: `Text Input • ${timestamp}`,
      isUser: true,
    });

    if (isTtsEnabled) {
      speakText(text);
    }

    // AI Response
    setTimeout(() => {
      const respTime = new Date().toLocaleTimeString("en-US", { hour12: false });
      addChatMessage({
        sender: "🤖 Counter #04 Interpreter",
        text: `Received: "${text}". Processing your request at Counter #04.`,
        meta: `Civic Terminal Assistant • ${respTime}`,
        isUser: false,
      });
      if (isTtsEnabled) {
        speakText(`Received: ${text}. Processing your request.`);
      }
    }, 400);

    chatInput.value = "";
  } else if (activeGestureObj) {
    commitGestureToChat(activeGestureObj);
  }
}

// =============================================================================
// 12. APPLICATION BOOTSTRAP
// =============================================================================
async function bootstrap() {
  console.log("[*] Initializing Dual-Hand (42 Points) Sign Language Terminal...");
  startKioskClock();
  initKioskEventListeners();
  renderVocabDeck();

  const visionReady = await initMediaPipeVision();
  if (visionReady) {
    const cameraReady = await startWebcam();
    if (cameraReady) {
      processLiveVideoFeed();
    }
  }
}

window.addEventListener("DOMContentLoaded", bootstrap);
