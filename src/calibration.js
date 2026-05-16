/**
 * FormForward Calibration Module
 *
 * Layer 2: Temporal synchronization between video pose data and Garmin telemetry.
 * Layer 3: Simple learned relationships (linear regression + nearest-neighbour)
 *          that map wearable telemetry patterns to pose-derived heuristics.
 *
 * The calibration profile is built during a short calibration run where both
 * video and Garmin data are available. During subsequent runs, the profile
 * lets the system infer likely form state from telemetry alone.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 2 — Temporal Synchronization
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Align video-derived heuristics with Garmin telemetry rows by timestamp.
 *
 * Strategy: for each video frame (with a known timestamp relative to the start
 * of the calibration video), find the closest Garmin telemetry row by absolute
 * time difference. The video start time is provided so we can map relative
 * video timestamps into the same epoch as the Garmin data.
 *
 * @param {Object} params
 * @param {Array}  params.videoHeuristics  - Per-frame heuristics from computeCalibrationHeuristics().perFrame
 * @param {Array}  params.telemetryRows    - Garmin telemetry rows (from FIT parser or CSV), each with { timeSeconds, cadence, verticalOscillation, groundContactTime, strideLength, pace, heartRate, gctBalance }
 * @param {number} params.videoStartOffsetSeconds - Offset (in seconds from run start) when the calibration video begins
 * @param {number} [params.maxGapSeconds=5] - Maximum allowed gap; pairs beyond this are discarded
 * @returns {{ pairs: Array, stats: Object }}
 */
export function synchronize({ videoHeuristics, telemetryRows, videoStartOffsetSeconds = 0, maxGapSeconds = 5 }) {
  if (!videoHeuristics?.length || !telemetryRows?.length) {
    return { pairs: [], stats: { matched: 0, discarded: 0, avgGapMs: null } };
  }

  const pairs = [];
  let discarded = 0;
  let totalGapMs = 0;

  for (const frame of videoHeuristics) {
    // Convert video-relative timestamp to run-relative timestamp
    const frameRunSeconds = videoStartOffsetSeconds + (frame.timestampMs / 1000);

    // Find the closest telemetry row by time
    let bestRow = null;
    let bestGap = Infinity;
    for (const row of telemetryRows) {
      const gap = Math.abs(row.timeSeconds - frameRunSeconds);
      if (gap < bestGap) {
        bestGap = gap;
        bestRow = row;
      }
    }

    if (!bestRow || bestGap > maxGapSeconds) {
      discarded++;
      continue;
    }

    totalGapMs += bestGap * 1000;
    pairs.push({
      timestampMs: frame.timestampMs,
      runSeconds: frameRunSeconds,
      gapMs: Math.round(bestGap * 1000),
      // Video-derived heuristics (pose)
      pose: {
        torsoLean: frame.torsoLean,
        verticalBounce: frame.verticalBounce,
        strideAsymmetry: frame.strideAsymmetry
      },
      // Garmin telemetry at the matched timestamp
      telemetry: {
        cadence: bestRow.cadence ?? null,
        verticalOscillation: bestRow.verticalOscillation ?? null,
        groundContactTime: bestRow.groundContactTime ?? null,
        strideLength: bestRow.strideLength ?? null,
        pace: bestRow.pace ?? null,
        heartRate: bestRow.heartRate ?? null,
        gctBalance: bestRow.gctBalance ?? null
      }
    });
  }

  return {
    pairs,
    stats: {
      matched: pairs.length,
      discarded,
      avgGapMs: pairs.length > 0 ? Math.round(totalGapMs / pairs.length) : null
    }
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
// Layer 3 — Simple Learned Relationships
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build a calibration profile from synchronized pairs.
 *
 * The profile contains:
 *   1. Linear regression coefficients mapping each telemetry metric to each
 *      pose heuristic (e.g., cadence -> torsoLean).
 *   2. The raw synchronized pairs stored as a lookup table for nearest-neighbour
 *      retrieval at inference time.
 *   3. Baseline statistics (means, standard deviations) for normalization.
 *
 * @param {Array} syncedPairs - Output of synchronize().pairs
 * @returns {Object} calibrationProfile
 */
export function buildCalibrationProfile(syncedPairs) {
  if (!syncedPairs || syncedPairs.length < 2) {
    return { valid: false, reason: "Need at least 2 synchronized pairs to build a profile." };
  }

  const telemetryKeys = ["cadence", "verticalOscillation", "groundContactTime", "strideLength", "pace", "heartRate", "gctBalance"];
  const poseKeys = ["torsoLean", "verticalBounce", "strideAsymmetry"];

  // Compute baselines
  const baselines = {};
  for (const key of [...telemetryKeys, ...poseKeys]) {
    const values = syncedPairs.map(p => {
      return telemetryKeys.includes(key) ? p.telemetry[key] : p.pose[key];
    }).filter(Number.isFinite);
    if (values.length === 0) {
      baselines[key] = { mean: null, std: null, min: null, max: null };
      continue;
    }
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    baselines[key] = {
      mean: round(mean, 2),
      std: round(Math.sqrt(variance), 2),
      min: round(Math.min(...values), 2),
      max: round(Math.max(...values), 2)
    };
  }

  // Linear regressions: for each (telemetry -> pose) pair, fit y = mx + b
  const regressions = {};
  for (const poseKey of poseKeys) {
    regressions[poseKey] = {};
    for (const telKey of telemetryKeys) {
      const validPairs = syncedPairs.filter(p =>
        Number.isFinite(p.telemetry[telKey]) && Number.isFinite(p.pose[poseKey])
      );
      if (validPairs.length < 2) {
        regressions[poseKey][telKey] = { slope: null, intercept: null, r2: null, n: 0 };
        continue;
      }
      const xs = validPairs.map(p => p.telemetry[telKey]);
      const ys = validPairs.map(p => p.pose[poseKey]);
      regressions[poseKey][telKey] = linearRegression(xs, ys);
    }
  }

  // Store the lookup table (for nearest-neighbour at inference time)
  const lookupTable = syncedPairs.map(p => ({
    telemetry: { ...p.telemetry },
    pose: { ...p.pose }
  }));

  return {
    valid: true,
    createdAt: new Date().toISOString(),
    sampleCount: syncedPairs.length,
    baselines,
    regressions,
    lookupTable,
    telemetryKeys,
    poseKeys
  };
}


/**
 * Infer pose heuristics from a single telemetry snapshot using the calibration profile.
 *
 * Uses a weighted blend of:
 *   1. Regression prediction (from the strongest correlated telemetry metric)
 *   2. Nearest-neighbour retrieval (Euclidean distance in normalized telemetry space)
 *
 * @param {Object} telemetrySnapshot - { cadence, verticalOscillation, groundContactTime, ... }
 * @param {Object} profile - Output of buildCalibrationProfile()
 * @param {Object} [options]
 * @param {number} [options.k=3] - Number of neighbours for kNN
 * @param {number} [options.regressionWeight=0.4] - Blend weight for regression (0-1)
 * @returns {{ inferred: Object, confidence: string, method: string, neighbours: Array, regressionPredictions: Object }}
 */
export function inferFromTelemetry(telemetrySnapshot, profile, options = {}) {
  if (!profile?.valid) {
    return { inferred: null, confidence: "none", method: "no_profile", neighbours: [], regressionPredictions: {} };
  }

  const k = options.k ?? 3;
  const regWeight = options.regressionWeight ?? 0.4;
  const knnWeight = 1 - regWeight;

  const { baselines, regressions, lookupTable, telemetryKeys, poseKeys } = profile;

  // ── Regression predictions ──
  const regressionPredictions = {};
  for (const poseKey of poseKeys) {
    // Pick the regression with the best R2 for this pose key
    let bestR2 = -1;
    let bestPrediction = null;
    let bestTelKey = null;
    for (const telKey of telemetryKeys) {
      const reg = regressions[poseKey]?.[telKey];
      if (!reg || reg.r2 === null || reg.slope === null) continue;
      const inputVal = telemetrySnapshot[telKey];
      if (!Number.isFinite(inputVal)) continue;
      if (reg.r2 > bestR2) {
        bestR2 = reg.r2;
        bestPrediction = reg.slope * inputVal + reg.intercept;
        bestTelKey = telKey;
      }
    }
    regressionPredictions[poseKey] = {
      value: bestPrediction !== null ? round(bestPrediction, 1) : null,
      r2: round(bestR2, 3),
      fromMetric: bestTelKey
    };
  }

  // ── Nearest-neighbour retrieval ──
  // Normalize the telemetry snapshot and each lookup entry, then compute Euclidean distance
  const distances = lookupTable.map((entry, idx) => {
    let sumSq = 0;
    let dims = 0;
    for (const key of telemetryKeys) {
      const inputVal = telemetrySnapshot[key];
      const entryVal = entry.telemetry[key];
      const bl = baselines[key];
      if (!Number.isFinite(inputVal) || !Number.isFinite(entryVal) || !bl || !bl.std || bl.std === 0) continue;
      const normInput = (inputVal - bl.mean) / bl.std;
      const normEntry = (entryVal - bl.mean) / bl.std;
      sumSq += (normInput - normEntry) ** 2;
      dims++;
    }
    return { index: idx, distance: dims > 0 ? Math.sqrt(sumSq / dims) : Infinity, entry };
  });

  distances.sort((a, b) => a.distance - b.distance);
  const neighbours = distances.slice(0, k).filter(d => d.distance < Infinity);

  // Weighted average of k nearest neighbours (inverse-distance weighting)
  const knnPredictions = {};
  for (const poseKey of poseKeys) {
    if (!neighbours.length) { knnPredictions[poseKey] = null; continue; }
    let weightSum = 0;
    let valueSum = 0;
    for (const n of neighbours) {
      const val = n.entry.pose[poseKey];
      if (!Number.isFinite(val)) continue;
      const w = n.distance > 0.001 ? 1 / n.distance : 1000; // avoid division by zero
      weightSum += w;
      valueSum += w * val;
    }
    knnPredictions[poseKey] = weightSum > 0 ? round(valueSum / weightSum, 1) : null;
  }

  // ── Blend regression + kNN ──
  const inferred = {};
  for (const poseKey of poseKeys) {
    const regVal = regressionPredictions[poseKey]?.value;
    const knnVal = knnPredictions[poseKey];
    if (Number.isFinite(regVal) && Number.isFinite(knnVal)) {
      inferred[poseKey] = round(regWeight * regVal + knnWeight * knnVal, 1);
    } else if (Number.isFinite(knnVal)) {
      inferred[poseKey] = knnVal;
    } else if (Number.isFinite(regVal)) {
      inferred[poseKey] = regVal;
    } else {
      inferred[poseKey] = null;
    }
  }

  // Confidence heuristic
  const avgR2 = Object.values(regressionPredictions)
    .map(p => p.r2)
    .filter(Number.isFinite);
  const meanR2 = avgR2.length > 0 ? avgR2.reduce((s, v) => s + v, 0) / avgR2.length : 0;
  const nearestDist = neighbours[0]?.distance ?? Infinity;
  let confidence = "low";
  if (meanR2 > 0.5 && nearestDist < 1.5) confidence = "high";
  else if (meanR2 > 0.25 || nearestDist < 2.0) confidence = "medium";

  return {
    inferred,
    confidence,
    method: `regression(w=${regWeight})+kNN(k=${k},w=${knnWeight})`,
    neighbours: neighbours.map(n => ({
      distance: round(n.distance, 3),
      telemetry: n.entry.telemetry,
      pose: n.entry.pose
    })),
    regressionPredictions
  };
}


/**
 * Compare inferred pose heuristics against the calibration baselines and
 * generate human-readable form change descriptions.
 *
 * @param {Object} inferred - Output of inferFromTelemetry().inferred
 * @param {Object} profile - Calibration profile
 * @returns {Array<{ metric: string, current: number, baseline: number, delta: number, direction: string, severity: string, description: string }>}
 */
export function detectFormChanges(inferred, profile) {
  if (!inferred || !profile?.valid) return [];

  const changes = [];
  const thresholds = {
    torsoLean: { warn: 3, bad: 6, unit: "deg", goodDesc: "Forward lean is stable", warnDesc: "Torso is leaning more than during calibration", badDesc: "Significant trunk collapse detected" },
    verticalBounce: { warn: 5, bad: 10, unit: "", goodDesc: "Vertical oscillation is consistent", warnDesc: "Slightly more bounce than calibration baseline", badDesc: "Excessive bouncing detected" },
    strideAsymmetry: { warn: 4, bad: 8, unit: "", goodDesc: "Stride symmetry is holding", warnDesc: "Mild asymmetry developing", badDesc: "Significant stride asymmetry detected" }
  };

  for (const [key, threshold] of Object.entries(thresholds)) {
    const current = inferred[key];
    const baseline = profile.baselines[key]?.mean;
    if (!Number.isFinite(current) || !Number.isFinite(baseline)) continue;

    const delta = round(current - baseline, 1);
    const absDelta = Math.abs(delta);
    let severity = "good";
    let description = threshold.goodDesc;
    if (absDelta >= threshold.bad) {
      severity = "bad";
      description = threshold.badDesc;
    } else if (absDelta >= threshold.warn) {
      severity = "warn";
      description = threshold.warnDesc;
    }

    changes.push({
      metric: key,
      current,
      baseline,
      delta,
      direction: delta > 0 ? "increased" : delta < 0 ? "decreased" : "stable",
      severity,
      description
    });
  }

  return changes;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Simple linear regression: y = slope * x + intercept
 * Returns { slope, intercept, r2, n }
 */
function linearRegression(xs, ys) {
  const n = xs.length;
  if (n < 2) return { slope: null, intercept: null, r2: null, n };

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
    sumXY += xs[i] * ys[i];
    sumX2 += xs[i] * xs[i];
    sumY2 += ys[i] * ys[i];
  }

  const denom = n * sumX2 - sumX * sumX;
  if (Math.abs(denom) < 1e-12) return { slope: 0, intercept: sumY / n, r2: 0, n };

  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  // R-squared
  const meanY = sumY / n;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < n; i++) {
    ssTot += (ys[i] - meanY) ** 2;
    ssRes += (ys[i] - (slope * xs[i] + intercept)) ** 2;
  }
  const r2 = ssTot > 0 ? Math.max(0, 1 - ssRes / ssTot) : 0;

  return { slope: round(slope, 6), intercept: round(intercept, 4), r2: round(r2, 4), n };
}

function round(v, d = 0) {
  if (!Number.isFinite(v)) return null;
  const f = 10 ** d;
  return Math.round(v * f) / f;
}
