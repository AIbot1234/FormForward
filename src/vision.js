/**
 * FormForward Vision Module
 * Inspired by:
 *  - henryczup/running-form-analyzer (head/trunk/shank angles, arm/hip swing, 3-tier grading)
 *  - JRKagumba/2D-video-based-running-analysis (gait event detection, stride metrics)
 *  - Ultralytics blog (YOLO pose estimation for running technique)
 *
 * MediaPipe Pose Landmarker → 33-point skeleton → biomechanical angle extraction
 * → form quality assessment → gait event classification → Gemma 4 VLM coaching
 */

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

let PoseLandmarker, FilesetResolver, DrawingUtils;
let landmarker = null;

// ─── Script loader ───

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.crossOrigin = "anonymous";
    s.onload = resolve;
    s.onerror = () => reject(new Error("Failed to load MediaPipe"));
    document.head.appendChild(s);
  });
}

export async function initVision() {
  try {
    await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js");
    const g = globalThis;
    FilesetResolver = g.FilesetResolver;
    PoseLandmarker = g.PoseLandmarker;
    DrawingUtils = g.DrawingUtils;
    if (!FilesetResolver || !PoseLandmarker) throw new Error("MediaPipe globals not found");
    const fileset = await FilesetResolver.forVisionTasks(WASM_CDN);
    landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "IMAGE",
      numPoses: 1
    });
    return true;
  } catch (err) {
    console.warn("Vision init failed, falling back to Gemma-only:", err.message);
    return false;
  }
}

export function isReady() { return !!landmarker; }

export function detectPose(imageEl) {
  if (!landmarker) return null;
  try {
    const result = landmarker.detect(imageEl);
    if (!result.landmarks || !result.landmarks.length) return null;
    return result.landmarks[0];
  } catch { return null; }
}

// ─── Landmark indices (MediaPipe 33-point model) ───

const LM = {
  NOSE: 0, L_EAR: 7, R_EAR: 8,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
  L_HEEL: 29, R_HEEL: 30,
  L_FOOT: 31, R_FOOT: 32
};

// ─── Geometry helpers ───

function angleDeg(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const cross = ab.x * cb.y - ab.y * cb.x;
  return Math.abs(Math.atan2(cross, dot) * (180 / Math.PI));
}

/** Angle of vector a→b from vertical (downward), in degrees. Positive = forward lean. */
function angleFromVertical(a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.atan2(dx, dy) * (180 / Math.PI);
}

function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2, visibility: Math.min(a.visibility || 0, b.visibility || 0) }; }

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// ─── Comprehensive angle extraction ───
// Modelled after henryczup/running-form-analyzer assessment metrics

export function computeAngles(lm) {
  if (!lm || lm.length < 33) return null;

  // Stricter validation: ensure at least some key landmarks are visible
  const visibleCount = lm.filter(pt => (pt.visibility || 0) > 0.3).length;
  if (visibleCount < 10) return null;

  const midShoulder = mid(lm[LM.L_SHOULDER], lm[LM.R_SHOULDER]);
  const midHip = mid(lm[LM.L_HIP], lm[LM.R_HIP]);
  const midAnkle = mid(lm[LM.L_ANKLE], lm[LM.R_ANKLE]);

  // 1. Head angle (ear → shoulder relative to vertical, à la running-form-analyzer)
  const headAngle = angleDeg(lm[LM.NOSE], midShoulder, midHip);

  // 2. Trunk / torso angle (forward lean from vertical)
  const trunkAngle = Math.abs(angleFromVertical(midShoulder, midHip));

  // 3. Knee angles (hip → knee → ankle)
  const leftKnee = angleDeg(lm[LM.L_HIP], lm[LM.L_KNEE], lm[LM.L_ANKLE]);
  const rightKnee = angleDeg(lm[LM.R_HIP], lm[LM.R_KNEE], lm[LM.R_ANKLE]);

  // 4. Elbow angles (shoulder → elbow → wrist)
  const leftElbow = angleDeg(lm[LM.L_SHOULDER], lm[LM.L_ELBOW], lm[LM.L_WRIST]);
  const rightElbow = angleDeg(lm[LM.R_SHOULDER], lm[LM.R_ELBOW], lm[LM.R_WRIST]);

  // 5. Shank angle at strike (knee → ankle vs vertical)
  const leftShank = Math.abs(angleFromVertical(lm[LM.L_KNEE], lm[LM.L_ANKLE]));
  const rightShank = Math.abs(angleFromVertical(lm[LM.R_KNEE], lm[LM.R_ANKLE]));

  // 6. Arm swing angle (shoulder → wrist relative to torso)
  const leftArmSwing = angleDeg(midHip, lm[LM.L_SHOULDER], lm[LM.L_WRIST]);
  const rightArmSwing = angleDeg(midHip, lm[LM.R_SHOULDER], lm[LM.R_WRIST]);

  // 7. Hip angle (knee → hip → shoulder, used for hip extension)
  const leftHipAngle = angleDeg(lm[LM.L_KNEE], lm[LM.L_HIP], lm[LM.L_SHOULDER]);
  const rightHipAngle = angleDeg(lm[LM.R_KNEE], lm[LM.R_HIP], lm[LM.R_SHOULDER]);

  // 8. Foot position relative to hip (overstride / understride detection)
  const leftOverstride = (lm[LM.L_ANKLE].x - lm[LM.L_HIP].x) * 100;
  const rightOverstride = (lm[LM.R_ANKLE].x - lm[LM.R_HIP].x) * 100;

  // 9. Hip drop (pelvic tilt — frontal plane)
  const hipDrop = Math.abs(lm[LM.L_HIP].y - lm[LM.R_HIP].y) * 100;

  // 10. Hip-ankle angle at strike (measures foot placement relative to CoM)
  const leftHipAnkle = Math.abs(angleFromVertical(lm[LM.L_HIP], lm[LM.L_ANKLE]));
  const rightHipAnkle = Math.abs(angleFromVertical(lm[LM.R_HIP], lm[LM.R_ANKLE]));

  // 11. Vertical oscillation proxy (nose Y relative to hip Y, normalized)
  const verticalOsc = Math.abs(lm[LM.NOSE].y - midHip.y) * 100;

  // 12. Stride width (lateral distance between ankles — balance metric)
  const strideWidth = Math.abs(lm[LM.L_ANKLE].x - lm[LM.R_ANKLE].x) * 100;

  return {
    headAngle: round(headAngle),
    trunkAngle: round(trunkAngle, 1),
    leftKnee: round(leftKnee), rightKnee: round(rightKnee),
    leftElbow: round(leftElbow), rightElbow: round(rightElbow),
    leftShank: round(leftShank, 1), rightShank: round(rightShank, 1),
    leftArmSwing: round(leftArmSwing), rightArmSwing: round(rightArmSwing),
    leftHipAngle: round(leftHipAngle), rightHipAngle: round(rightHipAngle),
    leftHipAnkle: round(leftHipAnkle, 1), rightHipAnkle: round(rightHipAnkle, 1),
    leftOverstride: round(leftOverstride, 1), rightOverstride: round(rightOverstride, 1),
    hipDrop: round(hipDrop, 1),
    verticalOsc: round(verticalOsc, 1),
    strideWidth: round(strideWidth, 1)
  };
}

// ─── 3-Tier assessment (Good / Needs Improvement / Bad) ───
// Thresholds from henryczup/running-form-analyzer assessment_calculator.py

const GRADE = { GOOD: "Good", IMPROVE: "Needs Improvement", BAD: "Bad" };

function gradeRange(val, badLow, improveLow, goodLow, goodHigh, improveHigh, badHigh) {
  if (val < badLow || val > badHigh) return GRADE.BAD;
  if (val < improveLow || val > improveHigh) return GRADE.IMPROVE;
  return GRADE.GOOD;
}

function gradeMetric(metric, val) {
  switch (metric) {
    case "headAngle":      return gradeRange(val, 0, 65, 80, 110, 120, 180);
    case "trunkAngle":     return gradeRange(val, -5, 1, 5, 15, 17, 30);
    case "leftElbow":
    case "rightElbow":     return gradeRange(val, 0, 53, 60, 90, 95, 180);
    case "leftShank":
    case "rightShank":     return val <= 10 ? GRADE.GOOD : val <= 15 ? GRADE.IMPROVE : GRADE.BAD;
    case "leftHipAnkle":
    case "rightHipAnkle":  return val <= 15 ? GRADE.GOOD : val <= 20 ? GRADE.IMPROVE : GRADE.BAD;
    case "hipDrop":        return val < 3 ? GRADE.GOOD : val < 5 ? GRADE.IMPROVE : GRADE.BAD;
    case "verticalOsc":    return val < 25 ? GRADE.GOOD : val < 35 ? GRADE.IMPROVE : GRADE.BAD;
    default:               return null;
  }
}

export function assessForm(angles) {
  if (!angles) return { score: null, issues: [], strengths: [], grades: {} };
  const issues = [];
  const strengths = [];
  const grades = {};

  // Grade each metric using the 3-tier system
  const metricAssess = [
    ["headAngle", "Head Position", angles.headAngle, "Keep gaze neutral — look 10-15m ahead, avoid chin drop or excessive lift"],
    ["trunkAngle", "Trunk / Forward Lean", angles.trunkAngle, "Maintain slight forward lean from ankles (5-15°) for efficient POSE running"],
    ["leftElbow", "Left Elbow", angles.leftElbow, "Keep elbows at ~60-90° — compact arm carry conserves energy"],
    ["rightElbow", "Right Elbow", angles.rightElbow, "Keep elbows at ~60-90° — compact arm carry conserves energy"],
    ["leftShank", "Left Shank Angle", angles.leftShank, "Vertical shin at foot strike minimises braking forces"],
    ["rightShank", "Right Shank Angle", angles.rightShank, "Vertical shin at foot strike minimises braking forces"],
    ["leftHipAnkle", "Left Foot Placement", angles.leftHipAnkle, "Land foot closer to center of mass — reduce overstriding"],
    ["rightHipAnkle", "Right Foot Placement", angles.rightHipAnkle, "Land foot closer to center of mass — reduce overstriding"],
    ["hipDrop", "Hip Stability", angles.hipDrop, "Strengthen glutes to maintain level pelvis during stance"],
    ["verticalOsc", "Vertical Oscillation", angles.verticalOsc, "Reduce bouncing — direct energy forward, not upward"]
  ];

  metricAssess.forEach(([key, label, value, recommendation]) => {
    const grade = gradeMetric(key, value);
    if (!grade) return;
    grades[key] = grade;
    if (grade === GRADE.GOOD) strengths.push(`${label}: ${value}° — ${grade}`);
    else issues.push({ label, value, grade, recommendation });
  });

  // Knee drive assessment (custom, not in the 3-tier table)
  const minKnee = Math.min(angles.leftKnee, angles.rightKnee);
  if (minKnee < 80) { strengths.push(`Active knee drive (${minKnee}°)`); grades.kneeDrive = GRADE.GOOD; }
  else if (minKnee > 165) { issues.push({ label: "Knee Drive", value: minKnee, grade: GRADE.BAD, recommendation: "Increase knee lift — shuffling gait wastes energy and shortens stride" }); grades.kneeDrive = GRADE.BAD; }
  else { grades.kneeDrive = GRADE.IMPROVE; }

  // Overstride assessment
  const maxOverstride = Math.max(angles.leftOverstride, angles.rightOverstride);
  if (maxOverstride > 8) { issues.push({ label: "Overstriding", value: round(maxOverstride, 1), grade: GRADE.BAD, recommendation: "Foot landing too far ahead of hips — increase cadence and shorten stride" }); grades.overstride = GRADE.BAD; }
  else if (maxOverstride > 5) { grades.overstride = GRADE.IMPROVE; }
  else { strengths.push("Foot landing under hips — efficient POSE alignment"); grades.overstride = GRADE.GOOD; }

  // Score: weighted by severity
  const goodCount = Object.values(grades).filter((g) => g === GRADE.GOOD).length;
  const badCount = Object.values(grades).filter((g) => g === GRADE.BAD).length;
  const total = Object.values(grades).length;
  const score = total > 0 ? Math.max(20, Math.min(100, Math.round((goodCount / total) * 100 - badCount * 5))) : null;

  return { score, issues, strengths, grades };
}

// ─── Skeleton drawing with angle annotations ───

const BODY_CONNECTIONS = [
  [11,13],[13,15],[12,14],[14,16],   // arms
  [11,12],[11,23],[12,24],[23,24],   // torso
  [23,25],[25,27],[24,26],[26,28],   // legs
  [27,29],[27,31],[28,30],[28,32],   // feet
  [0,11],[0,12]                       // head → shoulders
];

const JOINT_COLORS = {
  good: "#7ed6af", improve: "#f3a12b", bad: "#f36b6d"
};

export function drawSkeleton(canvas, landmarks, grades) {
  if (!landmarks || landmarks.length < 33) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();

  // Semi-transparent overlay for contrast
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.fillRect(0, 0, w, h);

  // Draw connections
  ctx.strokeStyle = "rgba(255, 155, 131, 0.8)";
  ctx.lineWidth = Math.max(2, w * 0.006);
  ctx.lineCap = "round";
  BODY_CONNECTIONS.forEach(([a, b]) => {
    const la = landmarks[a], lb = landmarks[b];
    if ((la.visibility || 0) > 0.3 && (lb.visibility || 0) > 0.3) {
      ctx.beginPath();
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
      ctx.stroke();
    }
  });

  // Draw keypoints with grade-based coloring
  const dotR = Math.max(4, w * 0.009);
  const gradeMap = grades || {};
  const jointGradeMap = {
    [LM.L_ELBOW]: gradeMap.leftElbow, [LM.R_ELBOW]: gradeMap.rightElbow,
    [LM.L_KNEE]: gradeMap.kneeDrive, [LM.R_KNEE]: gradeMap.kneeDrive,
    [LM.L_ANKLE]: gradeMap.leftHipAnkle, [LM.R_ANKLE]: gradeMap.rightHipAnkle,
    [LM.L_HIP]: gradeMap.hipDrop, [LM.R_HIP]: gradeMap.hipDrop,
    [LM.NOSE]: gradeMap.headAngle
  };

  landmarks.forEach((pt, i) => {
    if ((pt.visibility || 0) < 0.3) return;
    const grade = jointGradeMap[i];
    let color = "#fffaf4";
    if (grade === GRADE.GOOD) color = JOINT_COLORS.good;
    else if (grade === GRADE.IMPROVE) color = JOINT_COLORS.improve;
    else if (grade === GRADE.BAD) color = JOINT_COLORS.bad;
    else if (i === LM.NOSE) color = "#8f96ff";
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, dotR, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Draw angle annotations at key joints
  if (grades) drawAngleAnnotations(ctx, landmarks, w, h);

  ctx.restore();
}

function drawAngleAnnotations(ctx, lm, w, h) {
  ctx.font = `bold ${Math.max(10, w * 0.022)}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const annotations = [
    [LM.L_KNEE, () => round(angleDeg(lm[LM.L_HIP], lm[LM.L_KNEE], lm[LM.L_ANKLE])), "°", -14, 0],
    [LM.R_KNEE, () => round(angleDeg(lm[LM.R_HIP], lm[LM.R_KNEE], lm[LM.R_ANKLE])), "°", 14, 0],
    [LM.L_ELBOW, () => round(angleDeg(lm[LM.L_SHOULDER], lm[LM.L_ELBOW], lm[LM.L_WRIST])), "°", -14, -6],
    [LM.R_ELBOW, () => round(angleDeg(lm[LM.R_SHOULDER], lm[LM.R_ELBOW], lm[LM.R_WRIST])), "°", 14, -6]
  ];
  annotations.forEach(([idx, calcFn, suffix, ox, oy]) => {
    if ((lm[idx].visibility || 0) < 0.4) return;
    const val = calcFn();
    const tx = lm[idx].x * w + ox;
    const ty = lm[idx].y * h + oy;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const text = `${val}${suffix}`;
    const metrics = ctx.measureText(text);
    ctx.fillRect(tx - 2, ty - 8, metrics.width + 4, 16);
    ctx.fillStyle = "#fffaf4";
    ctx.fillText(text, tx, ty);
  });
}

// ─── Gait analysis (inspired by JRKagumba/2D-video-based-running-analysis) ───

export function analyzeGaitCycle(frameAnalyses) {
  if (frameAnalyses.length < 2) return null;
  const withAngles = frameAnalyses.filter((f) => f.angles);
  if (withAngles.length < 2) return null;

  // Detect gait phase from knee/ankle positions
  const phases = withAngles.map((f) => {
    const a = f.angles;
    const lStance = a.leftKnee > 150;
    const rStance = a.rightKnee > 150;
    const lSwing = a.leftKnee < 140;
    const rSwing = a.rightKnee < 140;
    let phase = "mid-stance";
    if (lStance && rSwing) phase = "left-stance / right-swing";
    else if (rStance && lSwing) phase = "right-stance / left-swing";
    else if (lSwing && rSwing) phase = "flight";
    return { label: f.label, phase, knees: { left: a.leftKnee, right: a.rightKnee } };
  });

  // Form degradation: compare first vs last frame
  const first = withAngles[0].angles;
  const last = withAngles[withAngles.length - 1].angles;
  const degradation = {
    trunkLean: delta(last.trunkAngle, first.trunkAngle),
    kneeDrive: delta(Math.min(last.leftKnee, last.rightKnee), Math.min(first.leftKnee, first.rightKnee)),
    armCarry: delta(avg(last.leftElbow, last.rightElbow), avg(first.leftElbow, first.rightElbow)),
    hipStability: delta(last.hipDrop, first.hipDrop),
    verticalOsc: delta(last.verticalOsc, first.verticalOsc),
    headControl: delta(last.headAngle, first.headAngle)
  };

  // Symmetry analysis
  const symmetry = withAngles.map((f) => {
    const a = f.angles;
    return {
      label: f.label,
      kneeSymmetry: round(Math.abs(a.leftKnee - a.rightKnee)),
      elbowSymmetry: round(Math.abs(a.leftElbow - a.rightElbow)),
      hipAnkleSymmetry: round(Math.abs(a.leftHipAnkle - a.rightHipAnkle), 1),
      shankSymmetry: round(Math.abs(a.leftShank - a.rightShank), 1)
    };
  });

  return { phases, degradation, symmetry };
}

// ─── Payload builder for Gemma 4 VLM ───

export function buildVisionPayload(frameAnalyses) {
  if (!frameAnalyses || !frameAnalyses.length) return null;
  const gait = analyzeGaitCycle(frameAnalyses);
  return {
    cv_pipeline: "MediaPipe Pose Landmarker (33-point body landmark detection)",
    models_used: [
      "Pose Estimation: MediaPipe Pose Landmarker Lite (float16)",
      "Angle Analysis: 12-metric biomechanical extraction",
      "Form Classification: 3-tier grading (Good / Needs Improvement / Bad)",
      "Gait Cycle: Phase detection + symmetry + degradation tracking",
      "VLM Coaching: Gemma 4 vision-language synthesis"
    ],
    frames_analyzed: frameAnalyses.length,
    per_frame: frameAnalyses.map((f) => ({
      label: f.label,
      angles: f.angles,
      assessment: f.assessment ? { score: f.assessment.score, issues_count: f.assessment.issues.length, strengths_count: f.assessment.strengths.length, grades: f.assessment.grades } : null
    })),
    gait_analysis: gait,
    form_degradation: gait?.degradation || null
  };
}

// ─── Video-derived cadence estimation ───
// Detects foot-strike events by tracking ankle Y-position peaks across frames.
// Each local minimum in ankle Y (highest screen position = lowest real position)
// approximates a ground contact event. Two consecutive strikes = one stride.

/**
 * Estimate cadence (steps per minute) from a sequence of frame analyses.
 * @param {Array} frameAnalyses - Array of { label, landmarks, angles, timestampMs }
 * @param {number} videoFps - Frames per second of the source video
 * @returns {{ estimatedCadenceSpm: number|null, strikeTimes: number[], strideIntervals: number[] }}
 */
export function estimateVideoCadence(frameAnalyses, videoFps = 30) {
  const withLandmarks = frameAnalyses.filter(f => f.landmarks && f.landmarks.length >= 33);
  if (withLandmarks.length < 4) return { estimatedCadenceSpm: null, strikeTimes: [], strideIntervals: [] };

  // Track the lower of the two ankle Y positions per frame (higher Y = lower on screen = closer to ground)
  const ankleYSeries = withLandmarks.map((f, i) => {
    const lAnkle = f.landmarks[LM.L_ANKLE];
    const rAnkle = f.landmarks[LM.R_ANKLE];
    return {
      index: i,
      timestampMs: f.timestampMs ?? (i / videoFps) * 1000,
      maxAnkleY: Math.max(lAnkle.y, rAnkle.y), // highest Y = lowest point = foot strike
      leftAnkleY: lAnkle.y,
      rightAnkleY: rAnkle.y
    };
  });

  // Detect local maxima in ankle Y (foot strike = ankle at lowest real-world point = highest Y)
  const strikeTimes = [];
  for (let i = 1; i < ankleYSeries.length - 1; i++) {
    const prev = ankleYSeries[i - 1].maxAnkleY;
    const curr = ankleYSeries[i].maxAnkleY;
    const next = ankleYSeries[i + 1].maxAnkleY;
    if (curr > prev && curr > next) {
      strikeTimes.push(ankleYSeries[i].timestampMs);
    }
  }

  if (strikeTimes.length < 2) return { estimatedCadenceSpm: null, strikeTimes, strideIntervals: [] };

  // Compute inter-strike intervals
  const strideIntervals = [];
  for (let i = 1; i < strikeTimes.length; i++) {
    strideIntervals.push(strikeTimes[i] - strikeTimes[i - 1]);
  }

  // Average interval -> cadence
  const avgIntervalMs = strideIntervals.reduce((s, v) => s + v, 0) / strideIntervals.length;
  const estimatedCadenceSpm = avgIntervalMs > 0 ? round(60000 / avgIntervalMs) : null;

  return { estimatedCadenceSpm, strikeTimes, strideIntervals };
}

/**
 * Compute the 4 calibration heuristics from a set of frame analyses.
 * These are the core features that get aligned with Garmin telemetry.
 * @param {Array} frameAnalyses - Array of { angles, landmarks, timestampMs }
 * @param {number} videoFps - Frames per second
 * @returns {{ torsoLean: number|null, cadence: number|null, strideAsymmetry: number|null, verticalBounce: number|null, perFrame: Array }}
 */
export function computeCalibrationHeuristics(frameAnalyses, videoFps = 30) {
  const withAngles = frameAnalyses.filter(f => f.angles);
  if (!withAngles.length) return { torsoLean: null, cadence: null, strideAsymmetry: null, verticalBounce: null, perFrame: [] };

  // Per-frame heuristics
  const perFrame = withAngles.map((f, i) => {
    const a = f.angles;
    return {
      timestampMs: f.timestampMs ?? (i / videoFps) * 1000,
      torsoLean: a.trunkAngle,
      verticalBounce: a.verticalOsc,
      strideAsymmetry: round(Math.abs(a.leftKnee - a.rightKnee) + Math.abs(a.leftHipAnkle - a.rightHipAnkle), 1),
      kneeSymmetry: round(Math.abs(a.leftKnee - a.rightKnee)),
      hipAnkleSymmetry: round(Math.abs(a.leftHipAnkle - a.rightHipAnkle), 1)
    };
  });

  // Aggregate
  const avgTorso = perFrame.reduce((s, f) => s + (f.torsoLean || 0), 0) / perFrame.length;
  const avgBounce = perFrame.reduce((s, f) => s + (f.verticalBounce || 0), 0) / perFrame.length;
  const avgAsymmetry = perFrame.reduce((s, f) => s + (f.strideAsymmetry || 0), 0) / perFrame.length;

  // Cadence from video
  const cadenceResult = estimateVideoCadence(frameAnalyses, videoFps);

  return {
    torsoLean: round(avgTorso, 1),
    cadence: cadenceResult.estimatedCadenceSpm,
    strideAsymmetry: round(avgAsymmetry, 1),
    verticalBounce: round(avgBounce, 1),
    perFrame,
    cadenceDetail: cadenceResult
  };
}

// ─── Helpers ───

function round(v, d = 0) {
  if (!Number.isFinite(v)) return null;
  const f = 10 ** d;
  return Math.round(v * f) / f;
}

function delta(a, b) { return round((a || 0) - (b || 0), 1); }
function avg(a, b) { return (a + b) / 2; }
