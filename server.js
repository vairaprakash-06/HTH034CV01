/**
 * server.js - Fast Express Backend for Sign Language Gesture Classification
 * 
 * Features:
 * - Express server running on port 3000 with CORS enabled.
 * - POST /predict accepts a JSON payload of 63 hand coordinates (x, y, z * 21).
 * - Spawns a child process executing predict.py, piping coordinates via stdin.
 * - Reads Python stdout and returns the translated word and confidence to the frontend.
 */

import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_SCRIPT = path.join(__dirname, "predict.py");
const PYTHON_CMD = process.env.PYTHON_BIN || "python";

// Middleware
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// Root health check endpoint
app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "Sign Language Bridge API",
    endpoint: "POST /predict",
    port: PORT,
  });
});

/**
 * POST /predict
 * Accepts:
 *   { "landmarks": [x0, y0, z0, ..., x20, y20, z20] }
 *   or { "features": [...] }
 *   or direct array [...]
 */
app.post("/predict", (req, res) => {
  // Extract coordinates array from varied payload formats
  let coordinates = null;

  if (Array.isArray(req.body)) {
    coordinates = req.body;
  } else if (req.body && typeof req.body === "object") {
    coordinates = req.body.landmarks || req.body.features || req.body.coordinates;
  }

  // Validate presence and length of coordinates
  if (!coordinates || !Array.isArray(coordinates)) {
    return res.status(400).json({
      error: "Invalid request payload. Expected an array of 63 hand coordinates.",
      receivedType: typeof coordinates,
    });
  }

  if (coordinates.length !== 63) {
    return res.status(422).json({
      error: `Expected exactly 63 coordinates (21 landmarks x 3 axes), received ${coordinates.length}.`,
    });
  }

  // Spawn Python inference process
  const pyProcess = spawn(PYTHON_CMD, [PYTHON_SCRIPT], {
    cwd: __dirname,
    stdio: ["pipe", "pipe", "pipe"],
  });

  let stdoutData = "";
  let stderrData = "";

  // Capture stdout from Python
  pyProcess.stdout.on("data", (chunk) => {
    stdoutData += chunk.toString();
  });

  // Capture stderr for debugging
  pyProcess.stderr.on("data", (chunk) => {
    stderrData += chunk.toString();
  });

  // Handle process termination
  pyProcess.on("close", (code) => {
    if (code !== 0 && !stdoutData.trim()) {
      console.error(`[!] Python predict.py exited with code ${code}. Stderr:`, stderrData);
      return res.status(500).json({
        error: "Prediction failed inside Python inference script.",
        details: stderrData.trim(),
      });
    }

    try {
      const trimmed = stdoutData.trim();
      let responsePayload;

      // Try parsing JSON output from predict.py
      try {
        const parsed = JSON.parse(trimmed);
        responsePayload = {
          prediction: parsed.prediction || parsed.word || parsed.label || "—",
          confidence: parsed.confidence !== undefined ? parsed.confidence : 1.0,
          status: parsed.status,
        };
      } catch {
        // Fallback: predict.py printed plain string/word
        responsePayload = {
          prediction: trimmed || "—",
          confidence: 1.0,
        };
      }

      return res.json(responsePayload);
    } catch (err) {
      console.error("[!] Error parsing Python output:", err);
      return res.status(500).json({
        error: "Failed to parse inference result.",
        rawOutput: stdoutData,
      });
    }
  });

  // Handle spawn failure (e.g. Python executable not found)
  pyProcess.on("error", (err) => {
    console.error(`[!] Failed to spawn Python process ('${PYTHON_CMD}'):`, err);
    return res.status(500).json({
      error: `Failed to execute Python inference runtime ('${PYTHON_CMD}').`,
      message: err.message,
    });
  });

  // Write coordinates JSON to Python's stdin and close stream
  pyProcess.stdin.write(JSON.stringify(coordinates));
  pyProcess.stdin.end();
});

// Start Express server
const server = app.listen(PORT, () => {
  console.log("=".repeat(60));
  console.log(`  SIGN LANGUAGE BRIDGE - EXPRESS API SERVER`);
  console.log("=".repeat(60));
  console.log(`[*] Server running at    : http://localhost:${PORT}`);
  console.log(`[*] Inference endpoint   : POST http://localhost:${PORT}/predict`);
  console.log(`[*] Python interpreter   : ${PYTHON_CMD}`);
  console.log(`[*] Prediction script    : ${PYTHON_SCRIPT}`);
  console.log("=".repeat(60) + "\n");
});

export default app;
