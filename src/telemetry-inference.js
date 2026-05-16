/**
 * FormForward Telemetry Inference Module (Layer 4)
 *
 * Provides a real-time demo surface that:
 *   1. Simulates a live Garmin telemetry stream from a loaded FIT/CSV file
 *   2. Uses the calibration profile (from calibration.js) to infer pose
 *      heuristics from each telemetry snapshot
 *   3. Generates natural-language coaching cues via Gemma when form changes
 *      are detected
 *
 * This allows runners to receive practical feedback on their form without
 * needing a camera, coach, or lab setup during everyday training.
 */

import { inferFromTelemetry, detectFormChanges } from "./calibration.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Simulated Garmin Stream
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a simulated telemetry stream from an array of telemetry rows.
 * Emits one row at a time at a configurable playback speed.
 *
 * @param {Array} telemetryRows - Array of { timeSeconds, cadence, verticalOscillation, groundContactTime, strideLength, pace, heartRate, gctBalance }
 * @param {Object} [options]
 * @param {number} [options.speedMultiplier=10] - Playback speed (10x = 1 min of data plays in 6 seconds)
 * @param {Function} [options.onTick] - Called with (row, index, total) for each emitted row
 * @param {Function} [options.onComplete] - Called when the stream ends
 * @returns {{ start: Function, stop: Function, pause: Function, resume: Function, isRunning: Function }}
 */
export function createSimulatedStream(telemetryRows, options = {}) {
  const speed = options.speedMultiplier ?? 10;
  const onTick = options.onTick ?? (() => {});
  const onComplete = options.onComplete ?? (() => {});

  let currentIndex = 0;
  let timer = null;
  let running = false;
  let paused = false;

  function getIntervalMs(index) {
    if (index >= telemetryRows.length - 1) return 1000;
    const dt = (telemetryRows[index + 1].timeSeconds - telemetryRows[index].timeSeconds);
    return Math.max(50, (dt * 1000) / speed);
  }

  function tick() {
    if (!running || paused) return;
    if (currentIndex >= telemetryRows.length) {
      running = false;
      onComplete();
      return;
    }
    onTick(telemetryRows[currentIndex], currentIndex, telemetryRows.length);
    currentIndex++;
    if (currentIndex < telemetryRows.length) {
      timer = setTimeout(tick, getIntervalMs(currentIndex - 1));
    } else {
      running = false;
      onComplete();
    }
  }

  return {
    start() {
      if (running) return;
      currentIndex = 0;
      running = true;
      paused = false;
      tick();
    },
    stop() {
      running = false;
      paused = false;
      if (timer) clearTimeout(timer);
      timer = null;
    },
    pause() {
      paused = true;
      if (timer) clearTimeout(timer);
      timer = null;
    },
    resume() {
      if (!running || !paused) return;
      paused = false;
      tick();
    },
    isRunning() { return running && !paused; },
    getProgress() { return { current: currentIndex, total: telemetryRows.length }; }
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
// Real-time Inference Engine
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create an inference engine that processes a telemetry stream and produces
 * form assessments and coaching cues.
 *
 * @param {Object} calibrationProfile - Output of buildCalibrationProfile()
 * @param {Object} [options]
 * @param {number} [options.windowSize=5] - Number of recent snapshots to average for stability
 * @param {number} [options.cueIntervalSeconds=30] - Minimum seconds between coaching cues
 * @param {Function} [options.onFormUpdate] - Called with { inferred, changes, timestamp }
 * @param {Function} [options.onCoachingCue] - Called with { cue, changes, timestamp }
 * @returns {{ processSnapshot: Function, getHistory: Function, getLatest: Function, reset: Function }}
 */
export function createInferenceEngine(calibrationProfile, options = {}) {
  const windowSize = options.windowSize ?? 5;
  const cueInterval = (options.cueIntervalSeconds ?? 30) * 1000;
  const onFormUpdate = options.onFormUpdate ?? (() => {});
  const onCoachingCue = options.onCoachingCue ?? (() => {});

  const history = [];
  let lastCueTime = 0;

  function processSnapshot(telemetryRow) {
    const timestamp = Date.now();

    // Map telemetry row keys to the format expected by calibration.js
    const snapshot = {
      cadence: telemetryRow.cadence ?? null,
      verticalOscillation: telemetryRow.verticalOscillation ?? telemetryRow.vertical_oscillation_mm ?? null,
      groundContactTime: telemetryRow.groundContactTime ?? telemetryRow.ground_contact_time_ms ?? null,
      strideLength: telemetryRow.strideLength ?? telemetryRow.stride_length_m ?? null,
      pace: telemetryRow.pace ?? null,
      heartRate: telemetryRow.heartRate ?? telemetryRow.heart_rate ?? null,
      gctBalance: telemetryRow.gctBalance ?? telemetryRow.gct_balance ?? null
    };

    // Infer pose from telemetry
    const result = inferFromTelemetry(snapshot, calibrationProfile);
    const changes = detectFormChanges(result.inferred, calibrationProfile);

    const entry = {
      timestamp,
      timeSeconds: telemetryRow.timeSeconds,
      telemetry: snapshot,
      inferred: result.inferred,
      confidence: result.confidence,
      changes,
      method: result.method
    };
    history.push(entry);

    // Smooth: average the last N inferred values for stability
    const smoothed = smoothInferred(history.slice(-windowSize));

    onFormUpdate({ inferred: smoothed, changes, confidence: result.confidence, timestamp, raw: result });

    // Generate coaching cue if form issues detected and enough time has passed
    const significantChanges = changes.filter(c => c.severity === "bad" || c.severity === "warn");
    if (significantChanges.length > 0 && (timestamp - lastCueTime) > cueInterval) {
      lastCueTime = timestamp;
      const cue = generateCoachingCue(significantChanges, smoothed, snapshot);
      onCoachingCue({ cue, changes: significantChanges, timestamp });
    }

    return entry;
  }

  function smoothInferred(recentEntries) {
    if (!recentEntries.length) return {};
    const keys = Object.keys(recentEntries[0].inferred || {});
    const smoothed = {};
    for (const key of keys) {
      const vals = recentEntries.map(e => e.inferred?.[key]).filter(Number.isFinite);
      smoothed[key] = vals.length > 0
        ? Math.round((vals.reduce((s, v) => s + v, 0) / vals.length) * 10) / 10
        : null;
    }
    return smoothed;
  }

  return {
    processSnapshot,
    getHistory() { return [...history]; },
    getLatest() { return history[history.length - 1] || null; },
    reset() { history.length = 0; lastCueTime = 0; }
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
// Coaching Cue Generator (rule-based, pre-Gemma)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate a natural-language coaching cue from detected form changes.
 * This provides immediate feedback; Gemma can be called for richer narratives.
 *
 * @param {Array} changes - Output of detectFormChanges()
 * @param {Object} smoothedInferred - Smoothed inferred pose heuristics
 * @param {Object} telemetry - Current telemetry snapshot
 * @returns {string} coaching cue
 */
function generateCoachingCue(changes, smoothedInferred, telemetry) {
  const cues = [];

  for (const change of changes) {
    switch (change.metric) {
      case "torsoLean":
        if (change.delta > 0) {
          cues.push(`Your trunk lean has increased by ${Math.abs(change.delta)} degrees. Think "tall chest" and lean from the ankles, not the waist.`);
        } else {
          cues.push(`Your torso is more upright than your calibration baseline. A slight forward lean (5-15 degrees) helps with efficient POSE running.`);
        }
        break;
      case "verticalBounce":
        if (change.delta > 0) {
          cues.push(`Your vertical bounce has increased. Focus on directing energy forward, not upward. Think "quiet head, light feet."`);
        } else {
          cues.push(`Your vertical oscillation has dropped. Good if intentional, but check you are not shuffling.`);
        }
        break;
      case "strideAsymmetry":
        cues.push(`Your stride symmetry has shifted (delta: ${change.delta}). Check if one side feels tighter or if you are favouring a leg.`);
        break;
    }
  }

  // Add telemetry context if available
  if (telemetry.cadence && telemetry.cadence < 160) {
    cues.push(`Cadence is at ${Math.round(telemetry.cadence)} spm. Try to bring it closer to 170-180 for better turnover.`);
  }

  return cues.join(" ");
}


// ═══════════════════════════════════════════════════════════════════════════════
// Gemma Integration for Richer Coaching Narratives
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build a Gemma prompt for telemetry-only coaching.
 * This is sent to the Gemma endpoint for a richer, more contextual response.
 *
 * @param {Object} params
 * @param {Object} params.inferred - Inferred pose heuristics
 * @param {Array}  params.changes - Detected form changes
 * @param {Object} params.telemetry - Current telemetry snapshot
 * @param {Object} params.profile - Calibration profile (baselines)
 * @param {string} params.confidence - Inference confidence level
 * @returns {Object} Gemma-ready payload
 */
export function buildGemmaInferencePayload({ inferred, changes, telemetry, profile, confidence }) {
  return {
    app: "FormForward",
    mode: "telemetry_inference",
    guardrail: "Use cautious proxy-based wording. Do not diagnose injuries or claim certainty. These are inferred estimates from wearable data calibrated against a short video, not direct measurements.",
    inference_context: {
      method: "Calibration-based inference: video pose heuristics mapped to Garmin telemetry via linear regression + k-nearest-neighbour retrieval",
      confidence,
      calibration_samples: profile?.sampleCount ?? 0
    },
    current_telemetry: telemetry,
    inferred_form: inferred,
    detected_changes: changes.map(c => ({
      metric: c.metric,
      current: c.current,
      baseline: c.baseline,
      delta: c.delta,
      severity: c.severity,
      description: c.description
    })),
    calibration_baselines: profile?.baselines ?? {},
    requested_output: {
      format: "A concise coaching paragraph (3-5 sentences). Start with what the runner is doing well, then address the most important form change. End with one specific, actionable cue they can apply immediately."
    }
  };
}
