import { initVision, isReady as visionReady, detectPose, computeAngles, assessForm, drawSkeleton, buildVisionPayload, analyzeGaitCycle } from "./vision.js";
import { parseFitFile } from "./fit-parser.js";
import { synchronize, buildCalibrationProfile, inferFromTelemetry, detectFormChanges } from "./calibration.js";
import { createSimulatedStream, createInferenceEngine, buildGemmaInferencePayload } from "./telemetry-inference.js";
import { computeCalibrationHeuristics } from "./vision.js";

const METRICS = {
  cadence: { label: "Cadence", unit: "spm", color: "#ff9b83", aliases: ["cadence", "runcadence", "cadencespm", "spm"] },
  verticalOscillation: { label: "Vertical oscillation", unit: "mm", color: "#8f96ff", aliases: ["verticaloscillation", "verticaloscillationmm", "vo"] },
  groundContactTime: { label: "Ground contact time", unit: "ms", color: "#f3a12b", aliases: ["groundcontacttime", "groundcontacttimems", "gct"] },
  strideLength: { label: "Stride length", unit: "m", color: "#7ed6af", aliases: ["stridelength", "stridelengthm", "stride"] },
  pace: { label: "Pace", unit: "min/km", color: "#f36b6d", aliases: ["pace", "paceminkm"] },
  heartRate: { label: "Heart rate", unit: "bpm", color: "#ff6f91", aliases: ["heartrate", "hr", "bpm"] },
  elevation: { label: "Elevation", unit: "m", color: "#c7bfb7", aliases: ["elevation", "altitude", "elevationm"] },
  gctBalance: { label: "GCT balance", unit: "% left", color: "#75d0e8", aliases: ["gctbalance", "groundcontactbalance", "groundcontacttimebalance"] }
};
const METRIC_KEYS = Object.keys(METRICS);
const DEFAULT_MODEL = "gemma4:latest";
const DEFAULT_RESEARCH_URLS = [
  "https://posemethod.com/running/",
  "https://www8.garmin.com/manuals/webhelp/GUID-0221611A-992D-495E-8DED-1DD448F7A066/EN-US/GUID-62A09512-518A-424A-8491-FE2B80CD2091.html",
  "https://ai.google.dev/gemma/docs/capabilities/vision/video-understanding"
].join("\n");
const PDFJS_CDN = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs";
const PDFJS_WORKER_CDN = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs";
const SAMPLE_CSV = `time,cadence,vertical_oscillation_mm,ground_contact_time_ms,stride_length_m,pace,heart_rate,elevation,gct_balance
0:00,178,78,232,0.95,5:17,142,22,50.2
1:00,178,78,231,0.96,5:14,145,23,50.3
2:00,179,78,232,0.95,5:13,147,24,50.4
3:00,177,78,234,0.96,5:16,149,25,50.6
4:00,177,79,234,0.96,5:15,151,26,50.5
5:00,177,80,237,0.97,5:18,153,27,50.9
6:00,175,82,241,0.98,5:22,156,28,51.2
7:00,173,84,246,0.99,5:26,160,29,51.7
8:00,170,87,252,1.00,5:29,164,30,52.4
9:00,168,91,259,1.01,5:33,168,31,53.5
10:00,165,94,266,1.02,5:38,172,32,54.5
11:00,163,96,272,1.03,5:43,176,33,55.6
12:00,161,98,279,1.04,5:47,180,34,56.2
13:00,160,99,282,1.05,5:50,181,34,56.6`;

const state = {
  activeTab: "dashboard",
  activeMetric: "cadence",
  analysis: null,
  sourceName: "Sample CSV run",
  video: { name: "", previewUrl: "", frames: [], status: "No video loaded" },
  pdf: { name: "", previewUrl: "", extractPreview: "", status: "No PDF selected", diagnostics: [] },
  backend: { healthy: null, label: "Checking", detail: "Research endpoints are being checked." },
  research: { sources: [], status: "Idle", running: false, manualSources: [] },
  gemma: { status: "Idle", output: null, error: null, running: false },
  gps: {
    supported: typeof navigator !== "undefined" && "geolocation" in navigator,
    tracking: false,
    watchId: null,
    startedAt: null,
    points: [],
    distanceMeters: 0,
    status: "GPS idle",
    error: null,
    lastFix: null
  },
  vision: {
    ready: false,
    frameAnalyses: [],
    status: "Initializing",
    liveMode: false,
    liveLoopId: null,
    liveCooldownUntil: 0,
    lastLiveFrameAt: 0,
    liveLatest: null,
    liveGemma: { status: "Waiting for live data", narrative: "", running: false, error: null, lastUpdated: null }
  },
  chat: {
    history: [],
    running: false
  },
  player: {
    level: 1,
    xp: 0,
    nextLevelXp: 100,
    achievements: {
      "badge-video": false,
      "badge-gemma": false,
      "badge-pdf": false,
      "badge-perfect": false
    }
  },
  calibration: {
    profile: null,
    status: "No calibration profile loaded.",
    calibVideoFile: null,
    calibFitFile: null,
    syncedPairs: [],
    heuristics: null
  },
  inference: {
    stream: null,
    engine: null,
    telemetryRows: [],
    running: false,
    status: "Load a calibration profile first.",
    latestCue: ""
  }
};

let pdfjsLibPromise = null;

const el = {
  tabBar: qs("#tabBar"),
  fileInput: qs("#fileInput"),
  loadSampleButton: qs("#loadSampleButton"),
  loadSampleGpxButton: qs("#loadSampleGpxButton"),
  sourceName: qs("#sourceName"),
  summaryGrid: qs("#summaryGrid"),
  compactEventList: qs("#compactEventList"),
  compactCoachList: qs("#compactCoachList"),
  metricTabs: qs("#metricTabs"),
  chartTitle: qs("#chartTitle"),
  metricChart: qs("#metricChart"),
  legendRow: qs("#legendRow"),
  eventList: qs("#eventList"),
  videoInput: qs("#videoInput"),
  liveTrackingButton: qs("#liveTrackingButton"),
  videoPreview: qs("#videoPreview"),
  videoStatus: qs("#videoStatus"),
  frameStrip: qs("#frameStrip"),
  researchUrls: qs("#researchUrls"),
  researchButton: qs("#researchButton"),
  researchStatus: qs("#researchStatus"),
  researchList: qs("#researchList"),
  backendStatusBadge: qs("#backendStatusBadge"),
  backendStatusText: qs("#backendStatusText"),
  pdfEngineLabel: qs("#pdfEngineLabel"),
  urlScrapeState: qs("#urlScrapeState"),
  pdfState: qs("#pdfState"),
  manualSourceTitle: qs("#manualSourceTitle"),
  manualSourceUrl: qs("#manualSourceUrl"),
  manualSourceSummary: qs("#manualSourceSummary"),
  composerCharCount: qs("#composerCharCount"),
  addManualSourceButton: qs("#addManualSourceButton"),
  clearAllSourcesButton: qs("#clearAllSourcesButton"),
  sourceCountBadge: qs("#sourceCountBadge"),
  generateGemmaButton: qs("#generateGemmaButton"),
  gemmaModelInput: qs("#gemmaModelInput"),
  localGemmaPathInput: qs("#localGemmaPathInput"),
  localHubBadge: qs("#localHubBadge"),
  localGemmaBadge: qs("#localGemmaBadge"),
  apiSourceMode: qs("#apiSourceMode"),
  customApiBaseInput: qs("#customApiBaseInput"),
  gemmaStatus: qs("#gemmaStatus"),
  gemmaOutput: qs("#gemmaOutput"),
  coachList: qs("#coachList"),
  payloadOutput: qs("#payloadOutput"),
  copyPayloadButton: qs("#copyPayloadButton"),
  chatArea: qs("#chatArea"),
  chatInput: qs("#chatInput"),
  chatForm: qs("#chatForm"),
  chatWelcome: qs("#chatWelcome"),
  resetChatButton: qs("#resetChatButton"),
  availableMetrics: qs("#availableMetrics"),
  missingMetrics: qs("#missingMetrics"),
  analyzeVisionButton: qs("#analyzeVisionButton"),
  poseDetectionStatus: qs("#poseDetectionStatus"),
  angleAnalysisStatus: qs("#angleAnalysisStatus"),
  formClassStatus: qs("#formClassStatus"),
  visionPanel: qs("#visionPanel"),
  poseFrameStrip: qs("#poseFrameStrip"),
  anglePanel: qs("#anglePanel"),
  angleGrid: qs("#angleGrid"),
  assessmentPanel: qs("#assessmentPanel"),
  formScoreBadge: qs("#formScoreBadge"),
  assessmentColumns: qs("#assessmentColumns"),
  liveScoreChip: qs("#liveScoreChip"),
  liveScoreRingValue: qs("#liveScoreRingValue"),
  liveScoreCopy: qs("#liveScoreCopy"),
  bodyPointGrid: qs("#bodyPointGrid"),
  liveInsightList: qs("#liveInsightList"),
  liveGemmaStatus: qs("#liveGemmaStatus"),
  liveGemmaNarrative: qs("#liveGemmaNarrative"),
  gpsStatusChip: qs("#gpsStatusChip"),
  gpsDistanceValue: qs("#gpsDistanceValue"),
  gpsPaceValue: qs("#gpsPaceValue"),
  gpsAccuracyValue: qs("#gpsAccuracyValue"),
  gpsFixValue: qs("#gpsFixValue"),
  gpsCoords: qs("#gpsCoords"),
  gpsRoutePreview: qs("#gpsRoutePreview"),
  pdfInput: qs("#pdfInput"),
  analyzePdfButton: qs("#analyzePdfButton"),
  pdfMeta: qs("#pdfMeta"),
  pdfPreviewShell: qs("#pdfPreviewShell"),
  pdfPreviewFrame: qs("#pdfPreviewFrame"),
  pdfExtractPreview: qs("#pdfExtractPreview"),
  trainDataInput: qs("#trainDataInput"),
  startTrainingButton: qs("#startTrainingButton"),
  // Optimal form panel
  optimalFormSection: qs("#optimalFormSection"),
  optimalFormImage: qs("#optimalFormImage"),
  formAngleOverlays: qs("#formAngleOverlays"),
  formAdjustmentsPanel: qs("#formAdjustmentsPanel"),
  adjustmentList: qs("#adjustmentList"),
  adjustmentSource: qs("#adjustmentSource"),
  pipelineBadge: qs("#pipelineBadge"),
  runFullPipelineButton: qs("#runFullPipelineButton"),
  // Gamification
  playerLevelBadge: qs("#playerLevelBadge"),
  playerXpLabel: qs("#playerXpLabel"),
  playerXpFill: qs("#playerXpFill"),
  // Calibration & Inference
  calibVideoInput: qs("#calibVideoInput"),
  calibFitInput: qs("#calibFitInput"),
  calibVideoOffsetInput: qs("#calibVideoOffsetInput"),
  runCalibrationButton: qs("#runCalibrationButton"),
  calibrationStatus: qs("#calibrationStatus"),
  calibProfilePanel: qs("#calibProfilePanel"),
  calibProfileSummary: qs("#calibProfileSummary"),
  calibRegressionGrid: qs("#calibRegressionGrid"),
  inferenceFitInput: qs("#inferenceFitInput"),
  inferenceSpeedSelect: qs("#inferenceSpeedSelect"),
  startInferenceButton: qs("#startInferenceButton"),
  stopInferenceButton: qs("#stopInferenceButton"),
  inferenceStatus: qs("#inferenceStatus"),
  inferenceDashboard: qs("#inferenceDashboard"),
  inferTorsoLean: qs("#inferTorsoLean"),
  inferVerticalBounce: qs("#inferVerticalBounce"),
  inferStrideAsymmetry: qs("#inferStrideAsymmetry"),
  inferConfidence: qs("#inferConfidence"),
  inferChangesList: qs("#inferChangesList"),
  inferCoachingCues: qs("#inferCoachingCues"),
  inferProgressBar: qs("#inferProgressBar"),
  inferProgressLabel: qs("#inferProgressLabel")
};

el.gemmaModelInput.value = DEFAULT_MODEL;
if (el.localGemmaPathInput) el.localGemmaPathInput.value = "local-models/gemma4";

// Hydrate and Boot Local Gateway Polling
if (el.apiSourceMode) {
  el.apiSourceMode.value = localStorage.getItem("ff_api_mode") || "auto";
}
if (el.customApiBaseInput) {
  el.customApiBaseInput.value = localStorage.getItem("ff_custom_base") || "";
}
startNetworkStatusPolling();

// Initialize vision pipeline
initVision().then((ok) => {
  state.vision.ready = ok;
  const stages = document.querySelectorAll(".pipeline-stage");
  if (ok) {
    el.poseDetectionStatus.textContent = "MediaPipe Pose Landmarker ready";
    stages[0]?.classList.add("done");
  } else {
    el.poseDetectionStatus.textContent = "Unavailable — Gemma vision fallback";
    stages[0]?.classList.add("error");
  }
});
el.researchUrls.value = DEFAULT_RESEARCH_URLS;
el.fileInput.addEventListener("change", async ({ target }) => {
  const [file] = target.files;
  if (!file) return;
  if (file.name.toLowerCase().endsWith(".fit")) {
    const buffer = await file.arrayBuffer();
    loadFitActivity(buffer, file.name);
    uploadFileToBackend(file, "fit"); // Auto persist to backend Garmin FIT Data Library
  } else {
    loadActivity(await file.text(), file.name);
    if (file.name.toLowerCase().endsWith(".csv")) {
      uploadFileToBackend(file, "csv"); // Auto persist to backend Garmin library
    }
  }
});
el.loadSampleButton.addEventListener("click", () => loadActivity(SAMPLE_CSV, "Sample CSV run"));
el.loadSampleGpxButton.addEventListener("click", () => loadActivity(buildSampleGpx(), "Sample GPX run"));
el.liveTrackingButton?.addEventListener("click", toggleLiveTracking);
el.videoInput.addEventListener("change", async ({ target }) => {
  const [file] = target.files;
  if (file) {
    await handleVideoUpload(file);
    uploadFileToBackend(file, "video"); // Auto persist to backend WhatsApp Videos library
  }
});
el.pdfInput?.addEventListener("change", ({ target }) => {
  const [file] = target.files;
  handlePdfSelection(file || null);
});
el.analyzeVisionButton.addEventListener("click", runVisionAnalysis);
el.researchButton.addEventListener("click", gatherResearch);
el.analyzePdfButton.addEventListener("click", analyzePdf);
el.addManualSourceButton.addEventListener("click", addManualSource);
el.clearAllSourcesButton.addEventListener("click", clearDynamicSources);
el.manualSourceSummary?.addEventListener("input", updateComposerCount);
el.startTrainingButton.addEventListener("click", startTraining);
el.runFullPipelineButton?.addEventListener("click", runFullPipeline);
el.copyPayloadButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(JSON.stringify(buildReport(), null, 2));
  el.copyPayloadButton.textContent = "Copied";
  setTimeout(() => { el.copyPayloadButton.textContent = "Copy JSON"; }, 1200);
});
el.generateGemmaButton.addEventListener("click", generateGemma);
el.chatForm?.addEventListener("submit", submitChat);
el.resetChatButton?.addEventListener("click", resetChat);

// Calibration & Inference event listeners
el.calibVideoInput?.addEventListener("change", ({ target }) => {
  state.calibration.calibVideoFile = target.files[0] || null;
});
el.calibFitInput?.addEventListener("change", ({ target }) => {
  state.calibration.calibFitFile = target.files[0] || null;
});
el.runCalibrationButton?.addEventListener("click", runCalibration);
el.inferenceFitInput?.addEventListener("change", async ({ target }) => {
  const file = target.files[0];
  if (!file) return;
  try {
    if (file.name.toLowerCase().endsWith(".fit")) {
      const buffer = await file.arrayBuffer();
      const { rows } = parseFitFile(buffer);
      state.inference.telemetryRows = rows.map((r, i) => ({
        timeSeconds: i,
        cadence: r.cadence,
        verticalOscillation: r.vertical_oscillation_mm,
        groundContactTime: r.ground_contact_time_ms,
        strideLength: r.stride_length_m,
        pace: r.pace,
        heartRate: r.heart_rate,
        gctBalance: r.gct_balance
      }));
    } else {
      // CSV fallback: use the existing parser
      const text = await file.text();
      state.inference.telemetryRows = parseInferenceCsv(text);
    }
    state.inference.status = `Loaded ${state.inference.telemetryRows.length} telemetry rows from ${file.name}.`;
    if (state.calibration.profile) el.startInferenceButton.disabled = false;
    el.inferenceStatus.textContent = state.inference.status;
  } catch (err) {
    state.inference.status = `Error loading file: ${err.message}`;
    el.inferenceStatus.textContent = state.inference.status;
  }
});
el.startInferenceButton?.addEventListener("click", startInferenceDemo);
el.stopInferenceButton?.addEventListener("click", stopInferenceDemo);
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (el.chatInput && !state.chat.running) {
      el.chatInput.value = chip.getAttribute("data-prompt") || "";
      submitChat();
    }
  });
});

// Tab switching
el.tabBar.addEventListener("click", ({ target }) => {
  const tab = target.closest(".tab");
  if (!tab) return;
  switchTab(tab.dataset.tab);
});
// Deep-dive links on dashboard
document.querySelectorAll(".deep-dive-link").forEach((link) => {
  link.addEventListener("click", () => switchTab(link.dataset.goto));
});

window.switchTab = function switchTab(tabId) {
  state.activeTab = tabId;
  el.tabBar.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabId));
  document.querySelectorAll(".tab-pane").forEach((p) => {
    const isActive = p.dataset.pane === tabId;
    p.classList.toggle("active", isActive);
    if (isActive) p.style.animation = "none", p.offsetHeight, p.style.animation = "";
  });
  if (tabId === "analysis" && state.analysis) { renderMetricTabs(); renderChart(); }
}

initManualSources();
refreshBackendHealth();
updateComposerCount();
loadActivity(SAMPLE_CSV, "Sample CSV run");

function loadActivity(text, name) {
  const parsed = parseActivity(text, name);
  if (!parsed.rows.length) {
    window.alert("No usable activity samples were found.");
    return;
  }
  state.analysis = analyzeRun(parsed.rows);
  state.sourceName = `${name} (${parsed.type})`;
  state.gemma = { status: "Idle", output: null, error: null, running: false };
  if (!state.analysis.availableMetrics.includes(state.activeMetric)) state.activeMetric = state.analysis.availableMetrics[0] || "cadence";
  render();
}

function parseActivity(text, name) {
  if (name.toLowerCase().endsWith(".gpx") || text.trimStart().startsWith("<")) return { type: "GPX", rows: parseGpx(text) };
  return { type: "CSV", rows: parseCsv(text) };
}

function loadFitActivity(buffer, name) {
  try {
    const result = parseFitFile(buffer);
    if (!result.rows.length) {
      window.alert("No usable running records found in this FIT file.");
      return;
    }
    state.analysis = analyzeRun(result.rows);
    state.sourceName = `${name} (FIT — ${result.summary.totalRecords} records, ${result.summary.durationMinutes} min)`;
    state.gemma = { status: "Idle", output: null, error: null, running: false };
    if (!state.analysis.availableMetrics.includes(state.activeMetric)) state.activeMetric = state.analysis.availableMetrics[0] || "cadence";
    gainXp(25); // Bonus XP for FIT import
    render();
  } catch (err) {
    window.alert(`FIT parse error: ${err.message}`);
  }
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = splitCsvLine(lines.shift() || "");
  return lines.map((line) => Object.fromEntries(splitCsvLine(line).map((value, index) => [headers[index], value])));
}

function splitCsvLine(line) {
  const cells = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === "\"") quoted = !quoted;
    else if (char === "," && !quoted) {
      cells.push(value.trim());
      value = "";
    } else value += char;
  }
  cells.push(value.trim());
  return cells;
}

function parseGpx(text) {
  const points = [...text.matchAll(/<(?:(?:[\w-]+):)?trkpt\b([^>]*)>([\s\S]*?)<\/(?:(?:[\w-]+):)?trkpt>/gi)].map((match) => {
    const body = match[2];
    return {
      time: tag(body, "time"),
      elevation: number(tag(body, "ele")),
      heart_rate: firstNumber(body, ["hr", "heartRate"]),
      cadence: normalizeCadence(firstNumber(body, ["cad", "cadence"])),
      vertical_oscillation_mm: firstNumber(body, ["VerticalOscillation", "verticalOscillation"]),
      ground_contact_time_ms: firstNumber(body, ["GroundContactTime", "groundContactTime"]),
      stride_length_m: firstNumber(body, ["StrideLength", "strideLength"]),
      gct_balance: firstNumber(body, ["GroundContactBalance", "gctBalance"])
    };
  });
  return points;
}

function analyzeRun(rows) {
  const samples = normalizeRows(rows);
  const windows = buildWindows(samples);
  const baseline = {};
  const early = windows.slice(0, Math.min(4, windows.length));
  METRIC_KEYS.forEach((key) => { baseline[key] = avg(early.map((window) => window[key])); });
  const events = detectEvents(windows, baseline);
  const patterns = mapPatterns(events);
  const availableMetrics = METRIC_KEYS.filter((key) => samples.some((sample) => Number.isFinite(sample[key])));
  const summary = summarize(samples, events, patterns);
  return { samples, windows, baseline, events, patterns, availableMetrics, missingMetrics: METRIC_KEYS.filter((key) => !availableMetrics.includes(key)), summary, gemmaPayload: buildGemmaPayload(summary, events, patterns, availableMetrics) };
}

function normalizeRows(rows) {
  if (!rows.length) return [];
  const headers = Object.keys(rows[0]);
  const headerMap = {};
  headers.forEach((header) => {
    const normalized = header.toLowerCase().replace(/[^a-z0-9]/g, "");
    if (["time", "timestamp", "elapsedtime"].includes(normalized)) headerMap.time = header;
    METRIC_KEYS.forEach((key) => { if (METRICS[key].aliases.includes(normalized)) headerMap[key] = header; });
  });
  let firstDate = null;
  return rows.map((row, index) => {
    const parsedTime = parseTime(row[headerMap.time] ?? String(index), firstDate);
    if (parsedTime.date && !firstDate) firstDate = parsedTime.date;
    return Object.fromEntries([
      ["timeSeconds", parsedTime.seconds ?? index],
      ...METRIC_KEYS.map((key) => [key, key === "pace" ? parsePace(row[headerMap[key]]) : number(row[headerMap[key]])])
    ]);
  }).sort((a, b) => a.timeSeconds - b.timeSeconds);
}

function buildWindows(samples) {
  const groups = new Map();
  samples.forEach((sample) => {
    const bucket = Math.floor(sample.timeSeconds / 60);
    if (!groups.has(bucket)) groups.set(bucket, []);
    groups.get(bucket).push(sample);
  });
  return [...groups.entries()].map(([index, group]) => Object.fromEntries([
    ["startSeconds", index * 60],
    ["endSeconds", index * 60 + 60],
    ...METRIC_KEYS.map((key) => [key, avg(group.map((sample) => sample[key]))])
  ]));
}

function detectEvents(windows, baseline) {
  const events = [];
  windows.forEach((window) => {
    const checks = [
      ["cadence_collapse", "Cadence collapse", Number.isFinite(baseline.cadence - window.cadence) && baseline.cadence - window.cadence > 8, `${Math.round(baseline.cadence - window.cadence)} spm below baseline`],
      ["vertical_spike", "Vertical oscillation spike", pct(window.verticalOscillation, baseline.verticalOscillation) > 15, `${Math.round(pct(window.verticalOscillation, baseline.verticalOscillation))}% above baseline`],
      ["gct_drift", "Ground contact drift", pct(window.groundContactTime, baseline.groundContactTime) > 8 || Math.abs((window.gctBalance || 50) - 50) > 4, "Longer or asymmetric ground contact"],
      ["stride_overextension", "Stride overextension", pct(window.strideLength, baseline.strideLength) > 3 && baseline.cadence - window.cadence > 6, "Stride length rising while cadence falls"],
      ["fatigue_risk", "Fatigue form risk", pct(window.heartRate, baseline.heartRate) > 8 && baseline.cadence - window.cadence > 4, "Heart rate rising while rhythm fades"]
    ];
    checks.forEach(([type, label, active, signal]) => {
      if (active) events.push({ id: `event-${events.length + 1}`, type, label, startSeconds: window.startSeconds, endSeconds: window.endSeconds, severity: type === "stride_overextension" || type === "gct_drift" ? 3 : 2, signals: [signal] });
    });
  });
  return events;
}

function mapPatterns(events) {
  const patterns = [];
  if (events.some((event) => event.type === "stride_overextension" || event.type === "cadence_collapse")) {
    patterns.push({ title: "Probable overstriding", confidence: "likely", severity: 3, evidence: collectEvidence(events, ["stride_overextension", "cadence_collapse"]), cue: "Your metrics suggest you may be reaching ahead as fatigue builds. Keep the foot landing under the hip.", drill: "5 minutes of wall-fall pulls, then 3 x 90 seconds of relaxed quick cadence.", nextRunFocus: "Hold cadence within 4 spm of your early-run rhythm for the first 15 minutes." });
  }
  if (events.some((event) => event.type === "vertical_spike")) {
    patterns.push({ title: "Excess vertical bounce", confidence: "possible", severity: 2, evidence: collectEvidence(events, ["vertical_spike"]), cue: "Your metrics suggest more energy is moving upward than forward. Think quiet head and forward fall.", drill: "6 x 20 seconds quick-feet strides on flat ground.", nextRunFocus: "Check for quiet shoulders and light ground contact after every kilometer." });
  }
  if (events.some((event) => event.type === "gct_drift")) {
    patterns.push({ title: "Late pull from support", confidence: "possible", severity: 3, evidence: collectEvidence(events, ["gct_drift"]), cue: "Your metrics suggest your support foot may be staying down too long. Pull sooner instead of pushing back.", drill: "3 x 60 seconds in-place Pose pulls, then 60 seconds easy jog.", nextRunFocus: "When effort rises, cue pull-pull-pull for 30 seconds before changing pace." });
  }
  if (events.some((event) => event.type === "fatigue_risk")) {
    patterns.push({ title: "Fatigue load affecting form", confidence: "possible", severity: 2, evidence: collectEvidence(events, ["fatigue_risk"]), cue: "Your metrics suggest effort climbed while mechanics softened. Protect form before speed.", drill: "5 minutes cadence resets: 20 seconds brisk rhythm, 40 seconds relaxed jog.", nextRunFocus: "If heart rate rises sharply, reduce pace first and protect cadence second." });
  }
  return patterns;
}

function summarize(samples, events, patterns) {
  const durationSeconds = samples.length ? samples.at(-1).timeSeconds - samples[0].timeSeconds : 0;
  
  let score = "N/A";
  if (events.length > 0) {
    score = Math.max(35, Math.round(100 - Math.min(55, events.reduce((sum, event) => sum + event.severity * 4, 0))));
  } else if (samples.length > 10) {
    score = 100;
  }
  return {
    durationSeconds,
    distanceKm: estimateDistance(samples),
    averagePace: avg(samples.map((sample) => sample.pace)),
    averageCadence: avg(samples.map((sample) => sample.cadence)),
    formHealthScore: score,
    eventCount: events.length,
    coachingCardCount: patterns.length
  };
}

function buildGemmaPayload(summary, events, patterns, availableMetrics) {
  return {
    app: "FormForward",
    guardrail: "Use cautious proxy-based wording. Do not diagnose injuries or claim certainty from wearable data or video frames.",
    run_summary: {
      duration: formatDuration(summary.durationSeconds),
      distance_km: round(summary.distanceKm, 2),
      average_pace_min_per_km: round(summary.averagePace, 2),
      average_cadence_spm: round(summary.averageCadence),
      form_health_score: summary.formHealthScore
    },
    metric_coverage: { available: availableMetrics },
    breakdown_events: events.map((event) => ({ id: event.id, type: event.type, timestamp: formatDuration(event.startSeconds), severity: event.severity, signals: event.signals })),
    pose_mappings: patterns.map((pattern) => ({ title: pattern.title, confidence: pattern.confidence, severity: pattern.severity, evidence: pattern.evidence, requested_output: { correction_cue: pattern.cue, drill: pattern.drill, next_run_focus: pattern.nextRunFocus } }))
  };
}

function render() {
  el.sourceName.textContent = state.sourceName;
  renderSummary();
  renderCompactEvents();
  renderCompactCoach();
  renderMetricTabs();
  renderChart();
  renderEvents();
  renderCoach();
  renderVideo();
  renderResearch();
  renderGemma();
  renderPayload();
  renderCoverage();
  renderLiveVision();
  renderGps();
  renderPdf();
  renderPlayer();
}

function renderPlayer() {
  if (el.playerLevelBadge) el.playerLevelBadge.textContent = state.player.level;
  if (el.playerXpLabel) el.playerXpLabel.textContent = `${Math.round(state.player.xp)} / ${state.player.nextLevelXp} XP`;
  if (el.playerXpFill) el.playerXpFill.style.width = `${Math.min(100, (state.player.xp / state.player.nextLevelXp) * 100)}%`;
  
  Object.keys(state.player.achievements).forEach(id => {
    const badge = qs(`#${id}`);
    if (badge) {
      if (state.player.achievements[id]) {
        badge.classList.add("unlocked");
      } else {
        badge.classList.remove("unlocked");
      }
    }
  });
}

function gainXp(amount) {
  state.player.xp += amount;
  if (state.player.xp >= state.player.nextLevelXp) {
    state.player.level += 1;
    state.player.xp -= state.player.nextLevelXp;
    state.player.nextLevelXp = Math.floor(state.player.nextLevelXp * 1.5);
    // Simple level up animation
    if (el.playerLevelBadge) {
      el.playerLevelBadge.style.transform = "scale(1.5)";
      setTimeout(() => { el.playerLevelBadge.style.transform = "scale(1)"; }, 300);
    }
  }
  renderPlayer();
}

function unlockAchievement(id) {
  if (!state.player.achievements[id]) {
    state.player.achievements[id] = true;
    gainXp(50); // bonus XP for achievement
  }
}

function renderSummary() {
  const s = state.analysis.summary;
  if (s.formHealthScore !== "N/A" && s.formHealthScore >= 90) unlockAchievement("badge-perfect");
  gainXp(10); // Base XP for viewing summary
  const displayScore = s.formHealthScore === "N/A" ? "Waiting" : `${s.formHealthScore}/100`;
  el.summaryGrid.innerHTML = [["Duration", formatDuration(s.durationSeconds)], ["Distance", `${s.distanceKm.toFixed(2)} km`], ["Avg pace", `${formatMetric(s.averagePace, "pace")} /km`], ["Avg cadence", `${formatMetric(s.averageCadence, "cadence")} spm`], ["Form score", displayScore], ["Breakdowns", String(s.eventCount)]]
    .map(([label, value]) => `<article class="summary-card"><span>${label}</span><strong>${value}</strong></article>`).join("");
}

function renderCompactEvents() {
  const events = state.analysis.events.slice(0, 4);
  el.compactEventList.innerHTML = events.length
    ? events.map((event) => `<div class="compact-event-item"><span class="severity-pill severity-${event.severity}">S${event.severity}</span><strong>${event.label}</strong><time>${formatDuration(event.startSeconds)}</time></div>`).join("") + (state.analysis.events.length > 4 ? `<div class="compact-event-item" style="justify-content:center;color:var(--muted);font-weight:850;font-size:0.84rem;">+${state.analysis.events.length - 4} more — deep dive to see all</div>` : "")
    : `<p class="empty-state">No breakdown clusters detected.</p>`;
}

function renderCompactCoach() {
  const patterns = state.analysis.patterns.slice(0, 3);
  el.compactCoachList.innerHTML = patterns.length
    ? patterns.map((p) => `<div class="compact-coach-item"><span class="coach-severity">S${p.severity}</span><div class="coach-info"><h3>${p.title}</h3><p>${p.cue}</p></div><span class="coach-confidence">${p.confidence}</span></div>`).join("")
    : `<p class="empty-state">Run looked stable against the current POSE proxy rules.</p>`;
}

function renderMetricTabs() {
  el.metricTabs.innerHTML = state.analysis.availableMetrics.map((key) => `<button class="${key === state.activeMetric ? "active" : ""}" type="button" data-key="${key}"><span style="background:${METRICS[key].color}"></span>${METRICS[key].label}</button>`).join("");
  el.metricTabs.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    state.activeMetric = button.dataset.key;
    renderMetricTabs();
    renderChart();
  }));
}

function renderChart() {
  const key = state.activeMetric;
  const metric = METRICS[key];
  const windows = state.analysis.windows.filter((window) => Number.isFinite(window[key]));
  el.chartTitle.textContent = `${metric.label} (${metric.unit})`;
  el.metricChart.innerHTML = "";
  el.metricChart.setAttribute("viewBox", "0 0 720 340");
  if (!windows.length) return;
  const values = windows.map((window) => window[key]);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const spread = Math.max(2, (rawMax - rawMin) * 0.15);
  const yMin = rawMin - spread;
  const yMax = rawMax + spread;
  const chartLeft = 62;
  const chartRight = 686;
  const chartTop = 30;
  const chartHeight = 246;
  const chartBottom = chartTop + chartHeight;
  const maxTime = Math.max(60, windows.at(-1).endSeconds);
  const x = (time) => chartLeft + (time / maxTime) * (chartRight - chartLeft);
  const y = (value) => chartBottom - ((value - yMin) / Math.max(1, yMax - yMin)) * chartHeight;

  appendSvg("rect", { x: 0, y: 0, width: 720, height: 340, rx: 8, class: "chart-bg" });

  // Y-axis grid lines + value labels
  [0, 0.25, 0.5, 0.75, 1].forEach((ratio) => {
    const yPos = chartTop + ratio * chartHeight;
    const val = yMax - ratio * (yMax - yMin);
    appendSvg("line", { x1: chartLeft, y1: yPos, x2: chartRight, y2: yPos, class: "grid-line" });
    appendSvg("text", { x: chartLeft - 6, y: yPos + 4, class: "axis-label axis-y", "text-anchor": "end" }, formatAxisValue(val, key));
  });

  // X-axis time labels
  const totalMinutes = Math.ceil(maxTime / 60);
  const xStep = totalMinutes <= 5 ? 1 : totalMinutes <= 15 ? 2 : totalMinutes <= 30 ? 5 : 10;
  for (let m = 0; m <= totalMinutes; m += xStep) {
    const sec = m * 60;
    if (sec > maxTime) break;
    const xPos = x(sec);
    appendSvg("line", { x1: xPos, y1: chartBottom, x2: xPos, y2: chartBottom + 5, class: "grid-line" });
    appendSvg("text", { x: xPos, y: chartBottom + 18, class: "axis-label axis-x", "text-anchor": "middle" }, `${m}:00`);
  }

  // Event bands
  state.analysis.events.forEach((event) => appendSvg("rect", { x: x(event.startSeconds), y: chartTop, width: Math.max(8, x(event.endSeconds) - x(event.startSeconds)), height: chartHeight, class: `event-band severity-${event.severity}` }));

  // Data line + points
  appendSvg("path", { d: windows.map((window, index) => `${index ? "L" : "M"}${x(window.startSeconds + 30).toFixed(1)},${y(window[key]).toFixed(1)}`).join(" "), fill: "none", stroke: metric.color, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round" });
  windows.forEach((window) => appendSvg("circle", { cx: x(window.startSeconds + 30), cy: y(window[key]), r: 4, fill: metric.color }));
  el.legendRow.innerHTML = `<span><i style="background:${metric.color}"></i>${metric.label}</span><span><i class="severity-2"></i>Medium</span><span><i class="severity-3"></i>High</span>`;
}

function formatAxisValue(val, key) {
  if (key === "pace") return `${Math.floor(val)}:${String(Math.round((val % 1) * 60)).padStart(2, "0")}`;
  if (key === "strideLength") return val.toFixed(2);
  return String(Math.round(val));
}

function renderEvents() {
  el.eventList.innerHTML = state.analysis.events.length ? state.analysis.events.map((event) => `<article class="event-item"><div><span class="severity-pill severity-${event.severity}">S${event.severity}</span><strong>${event.label}</strong></div><time>${formatDuration(event.startSeconds)} - ${formatDuration(event.endSeconds)}</time><p>${event.signals.join(" / ")}</p></article>`).join("") : `<p class="empty-state">No breakdown clusters detected.</p>`;
}

function renderCoach() {
  el.coachList.innerHTML = state.analysis.patterns.length ? state.analysis.patterns.map((pattern) => `<article class="coach-card"><div class="coach-card-heading"><div><span>${pattern.confidence}</span><h3>${pattern.title}</h3></div><strong>S${pattern.severity}</strong></div><dl><div><dt>Cue</dt><dd>${pattern.cue}</dd></div><div><dt>Drill</dt><dd>${pattern.drill}</dd></div><div><dt>Next run</dt><dd>${pattern.nextRunFocus}</dd></div></dl></article>`).join("") : `<p class="empty-state">Run looked stable against the current POSE proxy rules.</p>`;
}

async function uploadFileToBackend(file, type) {
  try {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    await new Promise((resolve) => { reader.onload = resolve; });
    const base64 = reader.result.split(",")[1];
    const response = await fetch(getApiUrl("/api/upload"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, base64_data: base64, type })
    });
    const result = await response.json();
    if (result.ok) {
      console.log(`Uploaded ${file.name} to backend successfully.`);
      if ((type === "fit" || type === "csv") && typeof loadFitLibrary === "function") {
        loadFitLibrary();
      }
      if (type === "video" && typeof loadWhatsAppVideos === "function") {
        loadWhatsAppVideos();
      }
    }
  } catch (err) {
    console.warn("File upload backend replication failed:", err);
  }
}

async function handleVideoUpload(file) {
  stopLiveTracking({ preservePreview: false, silent: true });
  if (state.video.previewUrl) URL.revokeObjectURL(state.video.previewUrl);
  state.video = { name: file.name, previewUrl: URL.createObjectURL(file), frames: [], status: "Sampling frames" };
  renderVideo();
  try {
    state.video.frames = await extractVideoFrames(state.video.previewUrl);
    state.video.status = `${state.video.frames.length} frames ready from ${file.name}`;
    unlockAchievement("badge-video");
    gainXp(20);
    el.analyzeVisionButton.disabled = false;
  } catch (error) {
    state.video.status = `Video sampling failed: ${error.message}`;
  }
  renderVideo();
  renderPayload();
}

async function runVisionAnalysis() {
  if (!state.video.frames.length) {
    if (el.videoPreview.srcObject && !el.videoPreview.paused) {
      // Live Tracking is active, capture a frame
      const canvas = document.createElement("canvas");
      canvas.width = el.videoPreview.videoWidth || 640;
      canvas.height = el.videoPreview.videoHeight || 480;
      canvas.getContext("2d").drawImage(el.videoPreview, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.74);
      state.video.frames = [{ label: "Live Capture", dataUrl, base64: dataUrl.split(",")[1] }];
      renderVideo();
    } else {
      return;
    }
  }

  el.analyzeVisionButton.disabled = true;
  el.analyzeVisionButton.textContent = "Analyzing…";
  state.vision.frameAnalyses = [];
  const stages = document.querySelectorAll(".pipeline-stage");

  if (visionReady()) {
    el.poseDetectionStatus.textContent = "Detecting poses…";
    el.poseFrameStrip.innerHTML = "";
    el.visionPanel.hidden = false;

    for (const frame of state.video.frames) {
      const img = new Image();
      img.src = frame.dataUrl;
      await new Promise((r) => { img.onload = r; });
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const landmarks = detectPose(img);
      const angles = landmarks ? computeAngles(landmarks) : null;
      const assessment = angles ? assessForm(angles) : null;
      if (landmarks) drawSkeleton(canvas, landmarks, assessment?.grades || null);
      state.vision.frameAnalyses.push({ label: frame.label, landmarks, angles, assessment });
      const wrap = document.createElement("div");
      wrap.className = "pose-frame-wrap";
      wrap.appendChild(canvas);
      const lbl = document.createElement("span");
      lbl.className = "pose-frame-label";
      lbl.textContent = frame.label + (landmarks ? " · 33 pts" : " · no pose");
      wrap.appendChild(lbl);
      el.poseFrameStrip.appendChild(wrap);
    }
    el.poseDetectionStatus.textContent = `${state.vision.frameAnalyses.filter((f) => f.landmarks).length}/${state.video.frames.length} poses detected`;
    stages[0]?.classList.add("done");
  } else {
    el.poseDetectionStatus.textContent = "MediaPipe unavailable — using Gemma vision";
    stages[0]?.classList.add("error");
  }

  // Angle analysis
  const withAngles = state.vision.frameAnalyses.filter((f) => f.angles);
  if (withAngles.length) {
    el.angleAnalysisStatus.textContent = `${withAngles.length} frames · 12 metrics per frame`;
    stages[1]?.classList.add("done");
    renderAngleGrid(withAngles);
  } else {
    el.angleAnalysisStatus.textContent = "No pose data for angle computation";
  }

  // Form classification + gait analysis
  const withAssess = state.vision.frameAnalyses.filter((f) => f.assessment?.score);
  if (withAssess.length) {
    el.formClassStatus.textContent = "3-tier grading applied";
    stages[2]?.classList.add("done");
  } else {
    el.formClassStatus.textContent = "Insufficient data for classification";
  }
  renderFormAssessment(withAssess);

  el.analyzeVisionButton.textContent = "Re-analyze";
  el.analyzeVisionButton.disabled = false;
  state.vision.liveLatest = withAssess.at(-1) || withAngles.at(-1) || state.vision.frameAnalyses.at(-1) || null;
  if (state.vision.liveLatest) {
    updateLiveStateFromAnalysis(state.vision.liveLatest, { narrative: false });
  }
  renderPayload();
}

function renderAngleGrid(analyses) {
  el.anglePanel.hidden = false;
  const last = analyses[analyses.length - 1];
  const a = last.angles;
  const g = last.assessment?.grades || {};
  const gradeClass = (key) => g[key] === "Good" ? "good" : g[key] === "Bad" ? "bad" : g[key] ? "warn" : "";
  const cards = [
    ["Head", a.headAngle, "°", gradeClass("headAngle")],
    ["Trunk Lean", a.trunkAngle, "°", gradeClass("trunkAngle")],
    ["L Knee", a.leftKnee, "°", gradeClass("kneeDrive")],
    ["R Knee", a.rightKnee, "°", gradeClass("kneeDrive")],
    ["L Elbow", a.leftElbow, "°", gradeClass("leftElbow")],
    ["R Elbow", a.rightElbow, "°", gradeClass("rightElbow")],
    ["L Shank", a.leftShank, "°", gradeClass("leftShank")],
    ["R Shank", a.rightShank, "°", gradeClass("rightShank")],
    ["L Foot", a.leftHipAnkle, "°", gradeClass("leftHipAnkle")],
    ["R Foot", a.rightHipAnkle, "°", gradeClass("rightHipAnkle")],
    ["Hip Drop", a.hipDrop, "%", gradeClass("hipDrop")],
    ["Vert. Osc", a.verticalOsc, "", gradeClass("verticalOsc")]
  ];
  el.angleGrid.innerHTML = cards.map(([label, value, unit, cls]) =>
    `<div class="angle-card ${cls}"><span class="angle-label">${label}</span><span class="angle-value">${value ?? "—"}<span class="angle-unit">${unit}</span></span></div>`
  ).join("");
}

function renderFormAssessment(analyses) {
  el.assessmentPanel.hidden = false;
  if (!analyses || analyses.length === 0) {
    el.formScoreBadge.textContent = "N/A";
    el.formScoreBadge.className = "form-score-badge empty";
    el.assessmentColumns.innerHTML = `<div class="assessment-col"><h3>Awaiting Analysis</h3><p>Upload a video with clear pose visibility to generate a score.</p></div>`;
    return;
  }
  const allIssues = [];
  const allStrengths = [];
  analyses.forEach((a) => {
    a.assessment.issues.forEach((i) => { const key = typeof i === "string" ? i : i.label; if (!allIssues.find((x) => (typeof x === "string" ? x : x.label) === key)) allIssues.push(i); });
    a.assessment.strengths.forEach((s) => { if (!allStrengths.includes(s)) allStrengths.push(s); });
  });
  const avgScore = Math.round(analyses.reduce((s, a) => s + a.assessment.score, 0) / analyses.length);
  const tier = avgScore >= 75 ? "high" : avgScore >= 50 ? "mid" : "low";
  el.formScoreBadge.textContent = `${avgScore}/100`;
  el.formScoreBadge.className = `form-score-badge ${tier}`;

  const issueHtml = allIssues.length ? allIssues.map((i) => {
    if (typeof i === "string") return `<div class="assessment-item"><span class="assessment-icon">🔴</span>${escapeHtml(i)}</div>`;
    const gradeTag = i.grade === "Bad" ? "bad" : "warn";
    return `<div class="assessment-item"><span class="grade-tag ${gradeTag}">${i.grade}</span><div><strong>${escapeHtml(i.label)}</strong> (${i.value}°)<p class="rec-text">${escapeHtml(i.recommendation)}</p></div></div>`;
  }).join("") : '<div class="assessment-item"><span class="assessment-icon">✅</span>No issues detected</div>';

  const strengthHtml = allStrengths.length ? allStrengths.map((s) =>
    `<div class="assessment-item"><span class="assessment-icon">🟢</span>${escapeHtml(s)}</div>`
  ).join("") : '<div class="assessment-item"><span class="assessment-icon">—</span>Upload video for analysis</div>';

  // Gait analysis section
  const gait = analyzeGaitCycle(state.vision.frameAnalyses);
  let gaitHtml = "";
  if (gait) {
    const d = gait.degradation;
    const arrow = (v) => v > 0 ? "↑" : v < 0 ? "↓" : "→";
    const dCls = (v, invert) => { const bad = invert ? v < 0 : v > 0; return Math.abs(v) < 1 ? "stable" : bad ? "worse" : "better"; };
    gaitHtml = `<div class="gait-section">
      <h3>📊 Gait Analysis</h3>
      <div class="gait-phases">${gait.phases.map((p) => `<span class="gait-phase-tag">${p.label}: ${p.phase}</span>`).join("")}</div>
      <h4>Form Degradation (early → late)</h4>
      <div class="degradation-grid">
        <span class="deg-item ${dCls(d.trunkLean, false)}">Trunk ${arrow(d.trunkLean)} ${d.trunkLean}°</span>
        <span class="deg-item ${dCls(d.kneeDrive, false)}">Knee drive ${arrow(d.kneeDrive)} ${d.kneeDrive}°</span>
        <span class="deg-item ${dCls(d.armCarry, false)}">Arm carry ${arrow(d.armCarry)} ${d.armCarry}°</span>
        <span class="deg-item ${dCls(d.hipStability, false)}">Hip drop ${arrow(d.hipStability)} ${d.hipStability}</span>
        <span class="deg-item ${dCls(d.verticalOsc, false)}">Vert. osc ${arrow(d.verticalOsc)} ${d.verticalOsc}</span>
        <span class="deg-item ${dCls(d.headControl, false)}">Head ${arrow(d.headControl)} ${d.headControl}°</span>
      </div>
      ${gait.symmetry.length ? `<h4>L/R Symmetry</h4><div class="degradation-grid">${gait.symmetry.map((s) => `<span class="deg-item ${s.kneeSymmetry > 10 ? "worse" : "stable"}">Knee Δ${s.kneeSymmetry}° (${s.label})</span><span class="deg-item ${s.elbowSymmetry > 10 ? "worse" : "stable"}">Elbow Δ${s.elbowSymmetry}° (${s.label})</span>`).join("")}</div>` : ""}
    </div>`;
  }

  el.assessmentColumns.innerHTML =
    `<div class="assessment-col issues"><h3>⚠️ Issues Detected</h3>${issueHtml}</div>` +
    `<div class="assessment-col strengths"><h3>✅ Strengths</h3>${strengthHtml}</div>` +
    gaitHtml;
}

async function extractVideoFrames(url) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.src = url;
  await once(video, "loadedmetadata");
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 720 / video.videoWidth);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  const frames = [];
  for (const ratio of [0.2, 0.5, 0.8]) {
    video.currentTime = Math.min(video.duration - 0.05, Math.max(0.05, video.duration * ratio));
    await once(video, "seeked");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.74);
    frames.push({ label: `${Math.round(ratio * 100)}%`, dataUrl, base64: dataUrl.split(",")[1] });
  }
  return frames;
}

function renderVideo() {
  el.videoStatus.textContent = state.video.status;
  const hasLiveStream = Boolean(el.videoPreview?.srcObject);
  const hasPreviewAsset = Boolean(state.video.previewUrl);
  el.videoPreview.hidden = !(hasPreviewAsset || hasLiveStream || state.vision.liveMode);
  el.videoPreview.classList.toggle("live-camera-active", hasLiveStream || state.vision.liveMode);
  if (!hasLiveStream && state.video.previewUrl && el.videoPreview.src !== state.video.previewUrl) el.videoPreview.src = state.video.previewUrl;
  el.frameStrip.innerHTML = state.video.frames.map((frame) => `<figure><img src="${frame.dataUrl}" alt="Sampled video frame at ${frame.label}" /><figcaption>${frame.label}</figcaption></figure>`).join("");
}

async function toggleLiveTracking() {
  if (state.vision.liveMode) {
    stopLiveTracking({ preservePreview: false });
    return;
  }
  try {
    if (!window.isSecureContext) {
      throw new Error("Camera access needs a secure browser context. Reload the app at localhost and try again.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser session does not expose camera access. Open the app in a browser window that allows media devices.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    stopLiveTracking({ preservePreview: true, silent: true });
    if (state.video.previewUrl) {
      URL.revokeObjectURL(state.video.previewUrl);
      state.video.previewUrl = "";
    }
    state.video.frames = [];
    state.video.name = "Live camera";
    state.video.status = "Camera connected. Warming up live tracking…";
    el.videoPreview.autoplay = true;
    el.videoPreview.controls = false;
    el.videoPreview.srcObject = stream;
    el.videoPreview.hidden = false;
    await once(el.videoPreview, "loadedmetadata");
    await el.videoPreview.play();
    state.vision.liveMode = true;
    state.video.status = "Live tracking active";
    state.vision.liveGemma = { status: "Watching live posture", narrative: "", running: false, error: null, lastUpdated: null };
    state.vision.frameAnalyses = [];
    state.vision.liveLatest = null;
    el.analyzeVisionButton.disabled = false;
    el.liveTrackingButton.textContent = "Stop Live Tracking";
    startGpsTracking();
    switchTab("video");
    renderVideo();
    renderLiveVision();
    renderGps();
    startLiveLoop();
  } catch (err) {
    state.video.status = describeCameraError(err);
    state.vision.liveGemma.status = "Camera unavailable";
    state.vision.liveGemma.error = null;
    renderVideo();
    renderLiveVision();
  }
}

function stopLiveTracking({ preservePreview = false, silent = false } = {}) {
  if (state.vision.liveLoopId) {
    cancelAnimationFrame(state.vision.liveLoopId);
    state.vision.liveLoopId = null;
  }
  const stream = el.videoPreview?.srcObject;
  if (stream && typeof stream.getTracks === "function") {
    stream.getTracks().forEach((track) => track.stop());
  }
  if (el.videoPreview) {
    el.videoPreview.pause?.();
    el.videoPreview.controls = true;
    if (!preservePreview) {
      el.videoPreview.srcObject = null;
      el.videoPreview.classList.remove("live-camera-active");
      el.videoPreview.hidden = !state.video.previewUrl;
    }
  }
  state.vision.liveMode = false;
  state.vision.liveCooldownUntil = 0;
  state.vision.lastLiveFrameAt = 0;
  stopGpsTracking({ keepSummary: true });
  el.liveTrackingButton.textContent = "Start Live Tracking";
  if (!silent && !state.video.previewUrl) {
    state.video.status = "Live tracking stopped";
    renderVideo();
  }
}

function startGpsTracking() {
  stopGpsTracking({ keepSummary: false, silent: true });
  if (!navigator.geolocation) {
    state.gps = { ...state.gps, supported: false, tracking: false, status: "Geolocation unavailable in this browser.", error: "unavailable" };
    renderGps();
    return;
  }
  state.gps = {
    ...state.gps,
    supported: true,
    tracking: true,
    startedAt: Date.now(),
    points: [],
    distanceMeters: 0,
    status: "Requesting GPS permission…",
    error: null,
    lastFix: null
  };
  renderGps();
  state.gps.watchId = navigator.geolocation.watchPosition(handleGpsFix, handleGpsError, {
    enableHighAccuracy: true,
    maximumAge: 1000,
    timeout: 10000
  });
}

function stopGpsTracking({ keepSummary = true, silent = false } = {}) {
  if (state.gps.watchId != null && navigator.geolocation?.clearWatch) {
    navigator.geolocation.clearWatch(state.gps.watchId);
  }
  const nextStatus = keepSummary && state.gps.points.length
    ? `GPS summary saved (${state.gps.points.length} fixes)`
    : silent ? state.gps.status : "GPS idle";
  state.gps = {
    ...state.gps,
    tracking: false,
    watchId: null,
    status: nextStatus
  };
  renderGps();
}

function handleGpsFix(position) {
  const point = {
    lat: round(position.coords.latitude, 6),
    lon: round(position.coords.longitude, 6),
    accuracy: round(position.coords.accuracy, 1),
    speed: Number.isFinite(position.coords.speed) ? round(position.coords.speed, 2) : null,
    timestamp: position.timestamp
  };
  const previous = state.gps.points[state.gps.points.length - 1];
  let distanceMeters = state.gps.distanceMeters;
  const segmentMeters = previous ? haversineMeters(previous, point) : 0;
  if (previous && segmentMeters < 250 && (point.accuracy == null || point.accuracy <= 100)) {
    distanceMeters += segmentMeters;
  }
  const shouldAppend = !previous || segmentMeters >= 2 || state.gps.points.length < 2;
  const points = shouldAppend ? [...state.gps.points, point].slice(-80) : [...state.gps.points.slice(0, -1), point];
  state.gps = {
    ...state.gps,
    tracking: true,
    points,
    distanceMeters,
    lastFix: point,
    status: `GPS active (${points.length} fixes)`,
    error: null
  };
  renderGps();
}

function handleGpsError(error) {
  const message = error?.code === 1
    ? "Location permission blocked. Allow location access in Chrome to track your run route."
    : error?.code === 2
      ? "Unable to determine your location right now."
      : error?.code === 3
        ? "GPS fix timed out. Move to a clearer outdoor signal and retry."
        : error?.message || "GPS tracking failed.";
  state.gps = {
    ...state.gps,
    tracking: false,
    watchId: null,
    status: message,
    error: message
  };
  renderGps();
}

function renderGps() {
  if (!el.gpsStatusChip) return;
  const pace = averageGpsPace(state.gps);
  const tone = state.gps.error ? "bad" : state.gps.tracking ? "good" : state.gps.points.length ? "warn" : "";
  el.gpsStatusChip.textContent = state.gps.status;
  el.gpsStatusChip.className = `live-gemma-status${tone ? ` ${tone}` : ""}`;
  el.gpsDistanceValue.textContent = state.gps.distanceMeters ? `${(state.gps.distanceMeters / 1000).toFixed(2)} km` : "0.00 km";
  el.gpsPaceValue.textContent = pace ? `${formatMetric(pace, "pace")} /km` : "—";
  el.gpsAccuracyValue.textContent = state.gps.lastFix?.accuracy ? `${state.gps.lastFix.accuracy} m` : "—";
  el.gpsFixValue.textContent = String(state.gps.points.length);
  el.gpsCoords.textContent = state.gps.lastFix
    ? `${state.gps.lastFix.lat}, ${state.gps.lastFix.lon}`
    : state.gps.supported
      ? "Start live tracking in Chrome to collect route points."
      : "Geolocation is not available in this browser session.";
  renderGpsRoute();
}

function renderGpsRoute() {
  if (!el.gpsRoutePreview) return;
  const points = state.gps.points;
  if (points.length < 2) {
    el.gpsRoutePreview.innerHTML = `<div class="empty-state">Your live route will appear here once GPS fixes start arriving.</div>`;
    return;
  }
  const lats = points.map((point) => point.lat);
  const lons = points.map((point) => point.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const width = 320;
  const height = 168;
  const padding = 16;
  const spanLon = Math.max(0.0001, maxLon - minLon);
  const spanLat = Math.max(0.0001, maxLat - minLat);
  const path = points.map((point, index) => {
    const x = padding + ((point.lon - minLon) / spanLon) * (width - padding * 2);
    const y = height - padding - ((point.lat - minLat) / spanLat) * (height - padding * 2);
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const last = points[points.length - 1];
  const lastX = padding + ((last.lon - minLon) / spanLon) * (width - padding * 2);
  const lastY = height - padding - ((last.lat - minLat) / spanLat) * (height - padding * 2);
  el.gpsRoutePreview.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Live GPS route preview">
      <defs>
        <linearGradient id="gpsRouteGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#9bf7ff"></stop>
          <stop offset="100%" stop-color="#ff8fae"></stop>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="rgba(255,255,255,0.03)"></rect>
      <path d="${path}" fill="none" stroke="url(#gpsRouteGradient)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"></path>
      <circle cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="6" fill="#4af3ff" stroke="#120d12" stroke-width="2"></circle>
    </svg>
  `;
}

function startLiveLoop() {
  const step = async () => {
    if (!state.vision.liveMode) return;
    const video = el.videoPreview;
    const ready = video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
    const enoughTime = performance.now() - state.vision.lastLiveFrameAt >= 650;
    if (ready && enoughTime) {
      state.vision.lastLiveFrameAt = performance.now();
      const analysis = await analyzeCurrentVideoFrame(video);
      if (analysis) {
        state.vision.liveLatest = analysis;
        state.vision.frameAnalyses = [...state.vision.frameAnalyses.slice(-7), analysis];
        updateLiveStateFromAnalysis(analysis, { narrative: true });
      }
    }
    state.vision.liveLoopId = requestAnimationFrame(step);
  };
  state.vision.liveLoopId = requestAnimationFrame(step);
}

async function analyzeCurrentVideoFrame(video) {
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 720 / Math.max(1, video.videoWidth));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.72);
  const img = new Image();
  img.src = dataUrl;
  await new Promise((resolve) => { img.onload = resolve; });
  const landmarks = visionReady() ? detectPose(img) : null;
  const angles = landmarks ? computeAngles(landmarks) : null;
  const assessment = angles ? assessForm(angles) : null;
  if (landmarks) drawSkeleton(canvas, landmarks, assessment?.grades || null);
  return {
    label: `Live ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`,
    landmarks,
    angles,
    assessment,
    dataUrl,
    base64: dataUrl.split(",")[1],
    canvas
  };
}

function updateLiveStateFromAnalysis(analysis, { narrative } = { narrative: true }) {
  if (!analysis?.angles || !analysis?.assessment) {
    state.vision.liveGemma.status = "Pose not detected";
    renderLiveVision();
    return;
  }
  el.poseDetectionStatus.textContent = analysis.landmarks ? "Live pose detected (33 points)" : "Searching for pose…";
  el.angleAnalysisStatus.textContent = "Live angle stream updating";
  el.formClassStatus.textContent = "Live form grading active";
  renderAngleGrid([analysis]);
  renderFormAssessment([analysis]);
  renderLiveVision();
  // OCR dot overlay
  if (analysis.landmarks && typeof renderOcrDots === "function") {
    renderOcrDots(analysis.landmarks, analysis.assessment);
  }
  if (narrative) scheduleLiveGemmaNarrative();
}

function renderLiveVision() {
  const live = state.vision.liveLatest;
  const assessment = live?.assessment;
  const score = assessment?.score;
  const tone = !Number.isFinite(score) ? "" : score >= 75 ? "good" : score >= 50 ? "warn" : "bad";
  
  if (Number.isFinite(score)) {
    el.liveScoreChip.textContent = `${score}/100`;
    el.liveScoreChip.className = `live-score-chip ${tone}`;
  } else {
    el.liveScoreChip.textContent = "Awaiting Analysis";
    el.liveScoreChip.className = "live-score-chip empty";
  }
  el.liveScoreRingValue.textContent = Number.isFinite(score) ? score : "--";
  el.liveScoreRingValue.parentElement.style.setProperty("--ring-progress", String(Math.max(0, Math.min(100, score || 0))));
  el.liveScoreRingValue.parentElement.style.setProperty("--ring-color", tone === "good" ? "#7ed6af" : tone === "warn" ? "#f3a12b" : "#f36b6d");
  el.liveScoreCopy.textContent = live?.assessment
    ? buildLiveSummary(live)
    : "Start live tracking to compare your current running posture against the target POSE profile.";
  el.bodyPointGrid.innerHTML = live?.assessment
    ? buildBodyPointCards(live)
    : `<div class="empty-state">No live body-point analysis yet.</div>`;
  el.liveInsightList.innerHTML = live?.assessment
    ? buildLiveInsightItems(live)
    : `<div class="empty-state">We’ll list the joints and segments that need attention once pose landmarks are visible.</div>`;
  const gemmaTone = state.vision.liveGemma.error ? "bad" : state.vision.liveGemma.running ? "warn" : state.vision.liveGemma.lastUpdated ? "good" : "";
  el.liveGemmaStatus.textContent = state.vision.liveGemma.error || state.vision.liveGemma.status;
  el.liveGemmaStatus.className = `live-gemma-status${gemmaTone ? ` ${gemmaTone}` : ""}`;
  el.liveGemmaNarrative.textContent = state.vision.liveGemma.error || state.vision.liveGemma.narrative || "When landmarks begin streaming, Gemma will explain which body points are out of range and what to correct first.";
}

function buildLiveSummary(live) {
  const issues = live.assessment.issues.slice(0, 2).map((issue) => typeof issue === "string" ? issue : issue.label.toLowerCase());
  if (!issues.length) return "Your current frame is tracking close to the target profile. Keep the same rhythm and posture.";
  return `Right now the biggest drift is around ${issues.join(" and ")}. The panel below shows which body points need the first correction.`;
}

function buildBodyPointCards(live) {
  const profile = liveBodyProfile(live);
  return profile.map((item) => `
    <article class="body-point-card ${item.tone}">
      <div class="body-point-meta">
        <strong>${item.label}</strong>
        <span class="body-point-grade">${item.grade}</span>
      </div>
      <p>${escapeHtml(item.detail)}</p>
    </article>
  `).join("");
}

function buildLiveInsightItems(live) {
  const issues = live.assessment.issues;
  if (!issues.length) {
    return `<article class="live-insight-item good"><span class="live-insight-bullet">✓</span><div><strong>Good alignment</strong><p>No major body-point issues are showing in the current frame.</p></div></article>`;
  }
  return issues.slice(0, 5).map((issue) => {
    const label = typeof issue === "string" ? issue : issue.label;
    const recommendation = typeof issue === "string" ? "Stay relaxed and keep your foot landing under the hips." : issue.recommendation;
    const grade = typeof issue === "string" ? "Needs Improvement" : issue.grade;
    const tone = grade === "Bad" ? "bad" : "warn";
    return `<article class="live-insight-item ${tone}">
      <span class="live-insight-bullet">${tone === "bad" ? "!" : "~"}</span>
      <div>
        <strong>${escapeHtml(label)}</strong>
        <p>${escapeHtml(recommendation)}</p>
      </div>
    </article>`;
  }).join("");
}

function liveBodyProfile(live) {
  const angles = live.angles;
  const grades = live.assessment.grades || {};
  return [
    bodyPoint("Head", grades.headAngle, angles.headAngle, "Keep a neutral gaze with the chin quiet and the neck long.", 80, 110, "°"),
    bodyPoint("Trunk", grades.trunkAngle, angles.trunkAngle, "Lean from the ankles so momentum moves forward, not down at the waist.", 5, 15, "°"),
    bodyPoint("Arms", worstGrade([grades.leftElbow, grades.rightElbow]), avgPair(angles.leftElbow, angles.rightElbow), "Compact elbow angles keep the upper body from getting noisy.", 60, 90, "°"),
    bodyPoint("Foot Strike", worstGrade([grades.leftHipAnkle, grades.rightHipAnkle, grades.overstride]), avgPair(angles.leftHipAnkle, angles.rightHipAnkle), "Land the foot closer to the hips to reduce braking.", 0, 15, "°"),
    bodyPoint("Pelvis", grades.hipDrop, angles.hipDrop, "Keep the hips level through stance to avoid side-to-side collapse.", 0, 3, "%"),
    bodyPoint("Bounce", grades.verticalOsc, angles.verticalOsc, "Reduce excess vertical motion so energy goes forward.", 0, 25, "")
  ];
}

function bodyPoint(label, grade, value, guidance, targetLow, targetHigh, unit) {
  const tone = grade === "Good" ? "good" : grade === "Bad" ? "bad" : "warn";
  const measured = Number.isFinite(value) ? `${value}${unit}` : "—";
  const target = `${targetLow}-${targetHigh}${unit}`;
  return {
    label,
    grade: grade || "Waiting",
    tone,
    detail: `Current ${measured}. Target ${target}. ${guidance}`
  };
}

function worstGrade(values) {
  if (values.includes("Bad")) return "Bad";
  if (values.includes("Needs Improvement")) return "Needs Improvement";
  if (values.includes("Good")) return "Good";
  return null;
}

function avgPair(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) ? round((a + b) / 2, 1) : Number.isFinite(a) ? a : Number.isFinite(b) ? b : null;
}

async function scheduleLiveGemmaNarrative() {
  if (!state.vision.liveMode || state.vision.liveGemma.running || Date.now() < state.vision.liveCooldownUntil) return;
  const live = state.vision.liveLatest;
  if (!live?.angles || !live?.assessment) return;
  state.vision.liveGemma.running = true;
  state.vision.liveGemma.status = "Gemma 4 explaining current body points…";
  state.vision.liveCooldownUntil = Date.now() + 5000;
  renderLiveVision();

  const prompt = {
    task: "Explain the runner's current body-point issues in real time.",
    live_pose: {
      score: live.assessment.score,
      angles: live.angles,
      issues: live.assessment.issues.map((issue) => typeof issue === "string" ? { label: issue } : issue),
      strengths: live.assessment.strengths.slice(0, 4)
    },
    desired_pose: {
      head_angle_deg: "80-110",
      trunk_lean_deg: "5-15",
      elbow_angle_deg: "60-90",
      hip_ankle_angle_deg: "0-15",
      hip_drop_pct: "0-3",
      vertical_osc_proxy: "0-25"
    },
    instruction: "Return JSON only with one field named realtime_explanation. Mention which body points are not optimal and the first correction cue."
  };

  try {
    const body = await callGemmaEndpoint({
      model: el.gemmaModelInput.value.trim() || DEFAULT_MODEL,
      local_model_path: el.localGemmaPathInput?.value.trim() || "",
      provider: getGemmaProviderPreference(),
      messages: [
        { role: "user", content: `You are a concise running form coach. Use only the provided pose-analysis data. Return strict JSON.\n\n${JSON.stringify(prompt)}` }
      ],
      stream: false,
      format: "json",
      options: { temperature: 0.1, num_predict: 180 }
    });
    const parsed = safeJson(body.content);
    state.vision.liveGemma = {
      status: "Gemma 4 live explanation ready",
      narrative: cleanGemmaField(parsed.realtime_explanation) || fallbackLiveNarrative(live),
      running: false,
      error: null,
      lastUpdated: new Date().toISOString()
    };
  } catch (error) {
    state.vision.liveGemma = {
      status: "Gemma 4 live fallback",
      narrative: fallbackLiveNarrative(live),
      running: false,
      error: null,
      lastUpdated: new Date().toISOString()
    };
  }
  renderLiveVision();
}

function fallbackLiveNarrative(live) {
  const issues = live.assessment.issues
    .slice(0, 3)
    .map((issue) => typeof issue === "string" ? issue : `${issue.label.toLowerCase()} is off`)
    .join(", ");
  return issues
    ? `Current frame suggests ${issues}. First fix: keep the foot landing under the hips and hold a small forward lean from the ankles.`
    : "Current frame is staying close to target form. Keep the same cadence and relaxed upper body.";
}

async function gatherResearch() {
  state.research.running = true;
  state.research.status = "Gathering web context...";
  el.urlScrapeState.textContent = "Fetching";
  renderResearch();
  try {
    const response = await apiJson("/api/research", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ urls: el.researchUrls.value.split(/\n+/).map((url) => url.trim()).filter(Boolean) }) });
    const body = await response.json();
    const scraped = body.sources || [];
    state.research.sources = [...state.research.manualSources, ...scraped];
    state.research.running = false;
    state.research.status = `${scraped.filter((source) => source.ok).length}/${scraped.length} web sources gathered.`;
    state.backend = { healthy: true, label: "Online", detail: "URL scraper responded normally." };
    el.urlScrapeState.textContent = "Complete";
  } catch (error) {
    state.research.running = false;
    state.research.status = `Research unavailable: ${error.message}`;
    state.backend = { healthy: false, label: "Offline", detail: error.message };
    el.urlScrapeState.textContent = "Blocked";
  }
  renderResearch();
  renderPayload();
}

async function analyzePdf() {
  const file = el.pdfInput.files[0];
  if (!file) {
    window.alert("Please select a PDF document first.");
    return;
  }

  state.research.running = true;
  state.research.status = `Analyzing ${file.name} locally...`;
  el.pdfState.textContent = "Extracting";
  renderResearch();

  try {
    const localText = await extractPdfTextInBrowser(file);
    if (localText && localText.replace(/\s+/g, " ").trim().length > 80) {
      const preview = localText.slice(0, 1200);
      state.pdf.extractPreview = preview;
      state.pdf.diagnostics = ["Client-side text extraction", `Characters: ${localText.length}`];
      state.pdf.status = "Analyzed with Browser PDF.js";
      state.backend = { healthy: true, label: "Online", detail: "PDF was parsed locally in the browser." };
      el.pdfState.textContent = "Browser PDF.js";
      upsertResearchSource({
        url: file.name,
        ok: true,
        title: `PDF Document: ${file.name}`,
        summary: localText,
        source_type: "pdf_browser"
      });
    } else {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      await new Promise(r => reader.onload = r);
      const base64Data = reader.result.split(',')[1];

      const response = await apiJson("/api/scrape-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_name: file.name, base64_data: base64Data })
      });

      const body = await response.json();
      if (body.error) throw new Error(body.error);

      state.pdf.extractPreview = body.preview || body.text?.slice(0, 1200) || "No text extracted.";
      state.pdf.diagnostics = body.diagnostics || [];
      state.pdf.status = body.engine ? `Analyzed with ${body.engine}` : "PDF analyzed";
      state.backend = { healthy: true, label: "Online", detail: "PDF extraction endpoint responded normally." };
      el.pdfState.textContent = body.engine || "Complete";
      upsertResearchSource({
        url: file.name,
        ok: true,
        title: `PDF Document: ${file.name}`,
        summary: body.text || "Extracted biomechanical form data from PDF. Text is ready for ML ingestion.",
        source_type: "pdf_backend"
      });
    }

    state.research.running = false;
    state.research.status = `Collated ${state.research.sources.length} sources for training.`;
  } catch (error) {
    state.research.running = false;
    state.research.status = `PDF scraping failed: ${error.message}`;
    state.pdf.status = `PDF analysis failed: ${error.message}`;
    state.pdf.extractPreview = "We couldn't extract text from this PDF with the available local tools.";
    state.pdf.diagnostics = [];
    state.backend = { healthy: false, label: "Offline", detail: error.message };
    el.pdfState.textContent = "Blocked";
  }

  renderPdf();
  renderResearch();
  renderPayload();
}

function upsertResearchSource(source) {
  const matchIndex = state.research.sources.findIndex((item) => item.url === source.url && item.title === source.title);
  if (matchIndex >= 0) state.research.sources[matchIndex] = source;
  else state.research.sources.push(source);
}

function handlePdfSelection(file) {
  if (state.pdf.previewUrl) URL.revokeObjectURL(state.pdf.previewUrl);
  if (!file) {
    state.pdf = { name: "", previewUrl: "", extractPreview: "", status: "No PDF selected", diagnostics: [] };
    renderPdf();
    return;
  }
  state.pdf = {
    name: file.name,
    previewUrl: URL.createObjectURL(file),
    extractPreview: "",
    status: `${file.name} loaded. Press Analyze to extract text.`,
    diagnostics: [`Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`]
  };
  el.pdfState.textContent = "Ready";
  renderPdf();
}

function renderPdf() {
  if (el.pdfMeta) {
    const diagnostics = state.pdf.diagnostics.length ? ` • ${state.pdf.diagnostics.join(" • ")}` : "";
    el.pdfMeta.textContent = `${state.pdf.status}${diagnostics}`;
  }
  if (el.pdfPreviewShell && el.pdfPreviewFrame) {
    el.pdfPreviewShell.hidden = !state.pdf.previewUrl;
    if (state.pdf.previewUrl && el.pdfPreviewFrame.src !== state.pdf.previewUrl) {
      el.pdfPreviewFrame.src = state.pdf.previewUrl;
    }
  }
  if (el.pdfExtractPreview) {
    el.pdfExtractPreview.textContent = state.pdf.extractPreview || "Extracted text preview will appear here after analysis.";
  }
  if (el.pdfEngineLabel) {
    el.pdfEngineLabel.textContent = state.pdf.status.startsWith("Analyzed with ") ? state.pdf.status.replace("Analyzed with ", "") : (state.pdf.name ? el.pdfState.textContent || "Pending" : "Idle");
  }
}

function initManualSources() {
  try {
    const stored = localStorage.getItem("ff_manual_sources");
    state.research.manualSources = stored ? JSON.parse(stored) : [];
  } catch (e) {
    state.research.manualSources = [];
  }
  state.research.sources = [...state.research.manualSources];
}

function addManualSource() {
  const title = el.manualSourceTitle.value.trim();
  const url = el.manualSourceUrl.value.trim();
  const summary = el.manualSourceSummary.value.trim();
  
  if (!title || !summary) {
    window.alert("Please provide both a Title and Content/Summary for the manual source.");
    return;
  }
  
  const newSource = {
    url: url || "Browser Stored",
    ok: true,
    title: title,
    summary: summary,
    isManual: true
  };
  
  state.research.manualSources.push(newSource);
  localStorage.setItem("ff_manual_sources", JSON.stringify(state.research.manualSources));
  
  // Clear inputs
  el.manualSourceTitle.value = "";
  el.manualSourceUrl.value = "";
  el.manualSourceSummary.value = "";
  
  remergeSources();
  state.research.status = `Stored "${title}" in browser storage.`;
  renderResearch();
  renderPayload();
}

function deleteManualSource(idx) {
  const sourceToDelete = state.research.sources[idx];
  if (!sourceToDelete || !sourceToDelete.isManual) return;
  
  state.research.manualSources = state.research.manualSources.filter(
    (s) => !(s.title === sourceToDelete.title && s.url === sourceToDelete.url)
  );
  
  localStorage.setItem("ff_manual_sources", JSON.stringify(state.research.manualSources));
  
  remergeSources();
  state.research.status = `Removed "${sourceToDelete.title}" from browser storage.`;
  renderResearch();
  renderPayload();
}

function remergeSources() {
  const scraped = state.research.sources.filter((s) => !s.isManual);
  state.research.sources = [...state.research.manualSources, ...scraped];
}

function clearDynamicSources() {
  state.research.sources = [...state.research.manualSources];
  state.research.status = "Dynamic scraped sources cleared.";
  renderResearch();
  renderPayload();
}

async function startTraining() {
  const files = el.trainDataInput.files;
  if (!files.length && !state.research.sources.length) {
    window.alert("Please upload some Garmin CSV data or collate research sources first.");
    return;
  }
  // Use the unified pipeline instead
  await runFullPipeline();
}

/**
 * Unified Pipeline: PDF → PaddleOCR → Gemma 4 → Optimal Form Output
 * Seamlessly chains the PDF analyzer straight to Gemma 4 training.
 */
async function runFullPipeline() {
  const btn = el.runFullPipelineButton;
  const badge = el.pipelineBadge;

  // Update UI to "running" state
  btn.disabled = true;
  btn.textContent = "⏳ Running pipeline…";
  btn.className = "full-pipeline-btn running";
  badge.className = "pipeline-badge running";
  badge.querySelector("span:last-child").textContent = "Processing…";

  // Stage 0: Seamlessly scrape research URLs if provided but not yet loaded
  const urlsToScrape = (el.researchUrls?.value || "").split(/\n+/).map((url) => url.trim()).filter(Boolean);
  const alreadyHasWebSources = state.research.sources.some((s) => s.source_type === "web_article" || (!s.isManual && !s.source_type?.includes("pdf")));
  
  if (urlsToScrape.length > 0 && !alreadyHasWebSources) {
    badge.querySelector("span:last-child").textContent = "Scraping web sources…";
    try {
      const response = await apiJson("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urlsToScrape })
      });
      const body = await response.json();
      const scraped = body.sources || [];
      state.research.sources = [...state.research.manualSources, ...scraped];
      state.research.status = `${scraped.filter((s) => s.ok).length}/${scraped.length} web sources gathered automatically.`;
      renderResearch();
    } catch (err) {
      console.warn("Pipeline auto-scrape failed:", err);
    }
  }

  // Gather all available context
  const pdfFile = el.pdfInput?.files?.[0];
  let pdfData = null;

  // Stage 1: Read PDF if available
  if (pdfFile) {
    badge.querySelector("span:last-child").textContent = "Extracting PDF (PaddleOCR)…";
    try {
      const reader = new FileReader();
      reader.readAsDataURL(pdfFile);
      await new Promise(r => reader.onload = r);
      pdfData = {
        file_name: pdfFile.name,
        base64_data: reader.result.split(",")[1]
      };
    } catch (err) {
      console.warn("PDF read failed:", err);
    }
  }

  badge.querySelector("span:last-child").textContent = "Building Gemma 4 prompt…";

  // Build the unified payload
  const visionData = buildVisionPayload(state.vision.frameAnalyses);
  const payload = {
    pdf_data: pdfData,
    video_frames: state.video.frames.map(f => ({ base64: f.base64 })),
    run_analysis: state.analysis?.gemmaPayload || null,
    research_sources: state.research.sources,
    cv_analysis: visionData,
    model: el.gemmaModelInput.value.trim() || DEFAULT_MODEL,
    provider: getGemmaProviderPreference(),
    local_model_path: el.localGemmaPathInput?.value.trim() || ""
  };

  badge.querySelector("span:last-child").textContent = "Querying Gemma 4…";

  try {
    const response = await fetch(getApiUrl("/api/analyze-form"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (result.error) {
      throw new Error(result.error);
    }

    // Success — update the form adjustments panel
    const coaching = result.coaching || {};
    if (coaching.raw) {
      throw new Error("Gemma failed to return structured data. Try adjusting the prompt or model.");
    }
    renderOptimalFormOutput(coaching, result);

    // Also update the Gemma output on the AI Coach tab
    state.gemma = {
      status: `Generated by ${result.model_used || "Gemma 4"} (unified pipeline)`,
      output: normalizeGemmaOutput(JSON.stringify(coaching)),
      error: null,
      running: false
    };
    renderGemma();
    renderPayload();

    // Update pipeline badge to success
    btn.textContent = "✅ Pipeline complete — re-run anytime";
    btn.className = "full-pipeline-btn success";
    badge.className = "pipeline-badge success";
    const stageNames = (result.pipeline || []).filter(s => s.status === "success").map(s => s.stage).join(" → ");
    badge.querySelector("span:last-child").textContent = stageNames || "Complete";

  } catch (err) {
    // Error state
    btn.textContent = "❌ Pipeline failed — retry";
    btn.className = "full-pipeline-btn";
    badge.className = "pipeline-badge error";
    badge.querySelector("span:last-child").textContent = err.message;

    state.gemma = { status: "Pipeline failed", output: null, error: err.message, running: false };
    renderGemma();
  }

  btn.disabled = false;
  setTimeout(() => {
    btn.textContent = "⚡ Run Full Pipeline (PDF → Gemma 4 → Form)";
    btn.className = "full-pipeline-btn";
  }, 8000);
}

/**
 * Renders the optimal form output from the unified pipeline onto the dashboard.
 * Shows angle overlays on the runner image and populates the adjustments panel.
 */
function renderOptimalFormOutput(coaching, pipelineResult) {
  const overlays = el.formAngleOverlays;
  const listEl = el.adjustmentList;
  if (!overlays || !listEl) return;

  // Clear previous overlays
  overlays.innerHTML = "";

  // Render angle overlay tags on the runner image
  const optimalAngles = coaching.optimal_angles || {};
  const anglePositions = [
    { key: "trunk_lean", label: "Trunk", top: "30%", left: "42%" },
    { key: "knee_drive", label: "Knee", top: "62%", left: "55%" },
    { key: "elbow_angle", label: "Elbow", top: "38%", left: "22%" },
    { key: "head_angle", label: "Head", top: "12%", left: "48%" },
    { key: "hip_angle", label: "Hip", top: "48%", left: "44%" },
    { key: "shank_angle", label: "Shank", top: "75%", left: "52%" }
  ];

  anglePositions.forEach(({ key, label, top, left }, i) => {
    const angle = optimalAngles[key];
    if (!angle) return;
    const tag = document.createElement("div");
    tag.className = "angle-overlay-tag";
    tag.style.top = top;
    tag.style.left = left;
    tag.style.animationDelay = `${i * 80}ms`;

    // Determine grade
    const current = angle.current;
    const optimal = angle.optimal;
    if (current != null && optimal != null) {
      const diff = Math.abs(current - optimal);
      tag.classList.add(diff < 5 ? "good" : diff < 12 ? "warn" : "bad");
      tag.textContent = `${label}: ${current}° → ${optimal}°`;
    } else {
      tag.classList.add("good");
      tag.textContent = `${label}: ${optimal || "—"}°`;
    }
    overlays.appendChild(tag);
  });

  // Render form adjustments
  const adjustments = coaching.form_adjustments || [];
  if (adjustments.length) {
    const priorityIcons = { high: "🔴", medium: "🟡", low: "🟢" };
    listEl.innerHTML = adjustments.map(adj => {
      const priority = (adj.priority || "medium").toLowerCase();
      return `<div class="adjustment-item ${priority}">
        <div class="adj-icon">${priorityIcons[priority] || "🎯"}</div>
        <div class="adj-content">
          <strong>${escapeHtml(adj.body_part || "Form")}</strong>
          <span class="adj-angles">${escapeHtml(adj.current_issue || "")}</span>
          <p>${escapeHtml(adj.correction || "")}</p>
        </div>
      </div>`;
    }).join("");
  }

  // Update source label
  const sourceLabel = el.adjustmentSource;
  if (sourceLabel) {
    const parts = [];
    if (pipelineResult?.pdf_text_preview) parts.push("PDF research");
    if (state.video.frames.length) parts.push("video analysis");
    if (state.research.sources.length) parts.push("web research");
    sourceLabel.textContent = parts.length
      ? `Based on ${parts.join(" + ")} via ${pipelineResult?.model_used || "Gemma 4"}`
      : `Generated by ${pipelineResult?.model_used || "Gemma 4"}`;
  }
}

function renderResearch() {
  el.researchStatus.textContent = state.research.status;
  el.researchButton.disabled = state.research.running;
  el.researchButton.textContent = state.research.running ? "Scraping..." : "Scrape URLs";
  if (el.backendStatusText) el.backendStatusText.textContent = state.backend.label;
  if (el.backendStatusBadge) {
    el.backendStatusBadge.textContent = state.backend.healthy == null ? "Backend checking…" : state.backend.healthy ? "Backend online" : "Backend offline";
    el.backendStatusBadge.className = `research-status-chip${state.backend.healthy == null ? "" : state.backend.healthy ? " good" : " bad"}`;
  }
  
  if (el.sourceCountBadge) {
    el.sourceCountBadge.textContent = state.research.sources.length;
  }
  if (el.urlScrapeState && !state.research.running && el.urlScrapeState.textContent === "Fetching") el.urlScrapeState.textContent = "Ready";
  
  el.researchList.innerHTML = state.research.sources.map((source, index) => {
    const isManual = !!source.isManual;
    const badge = isManual 
      ? `<span style="display:inline-block; font-size:0.68rem; padding:2px 6px; border-radius:4px; background:rgba(0,229,255,0.15); color:var(--accent); font-weight:900; margin-bottom:6px;">💻 Browser Stored</span>`
      : source.source_type === "pdf_browser"
        ? `<span style="display:inline-block; font-size:0.68rem; padding:2px 6px; border-radius:4px; background:rgba(126,214,175,0.14); color:#7ed6af; font-weight:900; margin-bottom:6px;">📄 Browser PDF.js</span>`
        : source.source_type === "pdf_backend"
          ? `<span style="display:inline-block; font-size:0.68rem; padding:2px 6px; border-radius:4px; background:rgba(243,161,43,0.14); color:var(--amber); font-weight:900; margin-bottom:6px;">🧪 Backend fallback</span>`
          : `<span style="display:inline-block; font-size:0.68rem; padding:2px 6px; border-radius:4px; background:rgba(255,255,255,0.08); color:var(--muted); font-weight:800; margin-bottom:6px;">🌐 Local Web Extract</span>`;
      
    const deleteBtn = isManual 
      ? `<button type="button" class="delete-source-btn" data-index="${index}" style="min-height:24px; padding:0 8px; font-size:0.72rem; background:rgba(255,51,102,0.1); border:1px solid rgba(255,51,102,0.2); color:#ff9b83; cursor:pointer; float:right; border-radius:4px;">Delete</button>`
      : "";
      
    return `<article class="research-item ${source.ok ? "" : "muted"}" style="position:relative; overflow:hidden;">
      ${deleteBtn}
      <div style="display:flex; flex-direction:column; gap:4px; margin-top:2px;">
        ${badge}
        <strong>${escapeHtml(source.title || source.url)}</strong>
        <a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer" style="font-size:0.78rem; word-break:break-all;">${escapeHtml(compactUrl(source.url))}</a>
        <p style="white-space:pre-wrap; margin-top:8px; line-height:1.4;">${escapeHtml(source.summary || "")}</p>
      </div>
    </article>`;
  }).join("");
  
  // Attach event listeners to delete buttons
  el.researchList.querySelectorAll(".delete-source-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      deleteManualSource(idx);
    });
  });
}

function updateComposerCount() {
  if (!el.composerCharCount || !el.manualSourceSummary) return;
  const count = el.manualSourceSummary.value.trim().length;
  el.composerCharCount.textContent = `${count} chars`;
}

async function refreshBackendHealth() {
  try {
    const response = await apiJson("/api/health", { method: "GET" }, 1, 5000);
    const body = await response.json();
    state.backend = {
      healthy: !!body.ok,
      label: body.ok ? "Online" : "Offline",
      detail: body.ok ? "Local API is ready for scraping and PDF extraction." : "Local API is unavailable."
    };
  } catch (error) {
    state.backend = { healthy: false, label: "Offline", detail: error.message };
  }
  renderResearch();
}

async function loadPdfJs() {
  if (!pdfjsLibPromise) {
    pdfjsLibPromise = import(PDFJS_CDN).then((module) => {
      const lib = module.default || module;
      if (lib.GlobalWorkerOptions) lib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_CDN;
      return lib;
    });
  }
  return pdfjsLibPromise;
}

async function extractPdfTextInBrowser(file) {
  const pdfjs = await loadPdfJs();
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data }).promise;
  const chunks = [];
  for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
    const page = await doc.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => item.str || "")
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (pageText) chunks.push(`Page ${pageNumber}: ${pageText}`);
  }
  return chunks.join("\n\n");
}

function getApiUrl(path) {
  const mode = el.apiSourceMode?.value || localStorage.getItem("ff_api_mode") || "auto";
  let base = "";

  if (mode === "local-hub") {
    base = el.customApiBaseInput?.value || "http://localhost:5173";
  } else if (mode === "ollama") {
    // Standard APIs fallback to standard local gateway, Gemma routes to Ollama direct!
    base = "http://localhost:5173";
  } else if (mode === "auto") {
    if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
      base = "http://localhost:5173";
    }
  }
  
  if (base && (path.startsWith("/") || path.startsWith("data/"))) {
    const normalizedPath = path.startsWith("/") ? path : "/" + path;
    return base.replace(/\/$/, "") + normalizedPath;
  }
  return path;
}

function getGemmaProviderPreference() {
  const mode = el.apiSourceMode?.value || localStorage.getItem("ff_api_mode") || "auto";
  return mode === "ollama" ? "ollama" : "auto";
}

async function callGemmaEndpoint(payload) {
  const mode = el.apiSourceMode?.value || "auto";
  let url = "/api/gemma";
  let requestBody = { ...payload };
  
  if (mode === "local-hub") {
    const customBase = el.customApiBaseInput?.value || "http://localhost:5173";
    url = customBase.replace(/\/$/, "") + "/api/gemma";
  } else if (mode === "ollama") {
    const ollamaBase = el.customApiBaseInput?.value || "http://localhost:11434";
    url = ollamaBase.replace(/\/$/, "") + "/api/chat";
    requestBody = {
      model: payload.model || DEFAULT_MODEL,
      messages: payload.messages.map(m => {
        const n = { role: m.role, content: m.content };
        if (m.images) n.images = m.images;
        return n;
      }),
      stream: false
    };
    if (payload.format === "json") requestBody.format = "json";
    if (payload.options) {
      requestBody.options = {
        temperature: payload.options.temperature,
        num_predict: payload.options.num_predict
      };
    }
  } else if (mode === "auto") {
    if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
      url = "http://localhost:5173/api/gemma";
    }
  }
  
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody)
  });
  
  const rawText = await response.text();
  let data = {};
  try {
    data = JSON.parse(rawText);
  } catch {
    throw new Error("Invalid API response payload received.");
  }
  
  if (data.error) throw new Error(data.error);
  return {
    content: data.message?.content || data.response || ""
  };
}

async function startNetworkStatusPolling() {
  async function checkStatus() {
    const customBase = el.customApiBaseInput?.value || "http://localhost:5173";
    const nodeBase = customBase.replace(/\/$/, "");
    
    let nodeOk = false;
    let activeNodeBase = "http://localhost:5173";
    try {
      const resp = await fetch("http://localhost:5173/api/health", { signal: AbortSignal.timeout(1200) });
      nodeOk = resp.ok;
    } catch {
      if (nodeBase !== "http://localhost:5173") {
        try {
          const resp = await fetch(`${nodeBase}/api/health`, { signal: AbortSignal.timeout(1200) });
          nodeOk = resp.ok;
          if (nodeOk) activeNodeBase = nodeBase;
        } catch {}
      }
    }
    
    if (el.localHubBadge) {
      const dot = el.localHubBadge.querySelector(".badge-dot");
      if (nodeOk) {
        el.localHubBadge.style.background = "rgba(0, 230, 118, 0.15)";
        el.localHubBadge.style.borderColor = "var(--green)";
        el.localHubBadge.style.color = "var(--green)";
        el.localHubBadge.childNodes[2].textContent = " GATEWAY: ONLINE";
        if (dot) { dot.style.background = "var(--green)"; dot.style.boxShadow = "0 0 6px var(--green)"; }
      } else {
        el.localHubBadge.style.background = "rgba(255, 51, 102, 0.1)";
        el.localHubBadge.style.borderColor = "var(--red)";
        el.localHubBadge.style.color = "var(--red)";
        el.localHubBadge.childNodes[2].textContent = " GATEWAY: OFFLINE";
        if (dot) { dot.style.background = "var(--red)"; dot.style.boxShadow = "none"; }
      }
    }

    let gemmaOk = false;
    let gemmaText = "NODE: OFFLINE";
    try {
      const resp = await fetch(`${activeNodeBase}/api/gemma/status`, { signal: AbortSignal.timeout(1600) });
      if (resp.ok) {
        const status = await resp.json();
        const ollama = status.providers?.ollama;
        const repoLocal = status.providers?.repo_local;
        if (ollama?.ok && ollama.has_default_model) {
          gemmaOk = true;
          gemmaText = "GEMMA 4: READY (OLLAMA)";
        } else if (ollama?.ok) {
          gemmaOk = false;
          gemmaText = "OLLAMA ONLINE: PULL GEMMA 4";
        } else if (repoLocal?.ok) {
          gemmaOk = true;
          gemmaText = "GEMMA 4: READY (PYTHON)";
        } else if (nodeOk) {
          gemmaText = "GEMMA 4: MODEL MISSING";
        }
      }
    } catch {
      if ((el.apiSourceMode?.value || "auto") === "ollama") {
        try {
          const resp = await fetch("http://localhost:11434/api/tags", { signal: AbortSignal.timeout(1200) });
          gemmaOk = resp.ok;
          gemmaText = resp.ok ? "GEMMA 4: OLLAMA DIRECT" : "GEMMA 4: OFFLINE";
        } catch {}
      } else if (nodeOk) {
        gemmaOk = false;
        gemmaText = "GEMMA 4: GATEWAY ONLINE";
      }
    }

    if (el.localGemmaBadge) {
      const dot = el.localGemmaBadge.querySelector(".badge-dot");
      if (gemmaOk) {
        el.localGemmaBadge.style.background = "rgba(0, 229, 255, 0.15)";
        el.localGemmaBadge.style.borderColor = "var(--accent)";
        el.localGemmaBadge.style.color = "var(--accent)";
        el.localGemmaBadge.childNodes[2].textContent = " " + gemmaText;
        if (dot) { dot.style.background = "var(--accent)"; dot.style.boxShadow = "0 0 6px var(--accent)"; }
      } else {
        el.localGemmaBadge.style.background = "rgba(255, 255, 255, 0.05)";
        el.localGemmaBadge.style.borderColor = "rgba(255, 255, 255, 0.15)";
        el.localGemmaBadge.style.color = "var(--muted)";
        el.localGemmaBadge.childNodes[2].textContent = " GEMMA 4: OFFLINE";
        if (dot) { dot.style.background = "var(--muted)"; dot.style.boxShadow = "none"; }
      }
    }
  }
  checkStatus();
  setInterval(checkStatus, 6000);
  window.addEventListener("api-mode-change", checkStatus);
}

async function apiJson(path, options = {}, retries = 1, timeoutMs = 15000) {
  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(getApiUrl(path), { ...options, signal: controller.signal });
      clearTimeout(timeout);
      if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try {
          const body = await response.clone().json();
          if (body.error) message = body.error;
        } catch {}
        throw new Error(message);
      }
      return response;
    } catch (error) {
      clearTimeout(timeout);
      lastError = error;
      if (attempt < retries) await new Promise((resolve) => setTimeout(resolve, 350));
    }
  }
  if (lastError?.name === "AbortError") {
    throw new Error("The local backend timed out. Restart the dev server and try again.");
  }
  throw new Error(lastError?.message?.includes("Failed to fetch")
    ? "The local backend is not reachable. Restart the dev server at http://localhost:5173."
    : lastError?.message || "Unknown backend error.");
}

async function generateGemma() {
  state.gemma = { status: "Connecting to Gemma 4", output: null, error: null, running: true };
  renderGemma();
  const report = buildReport();
  const system = "You are FormForward, a cautious POSE running coach. Use proxy-based wording. You receive wearable metrics and computer vision analysis (MediaPipe Pose Landmarker skeleton detection with biomechanical angle computation). Return only JSON with correction_cue, drill, next_run_focus, visual_observations, research_notes. For visual_observations, synthesize the CV pipeline findings with what you see in the frames.";
  const userMessage = { role: "user", content: `${system}\n\nCreate one concise coaching response from this payload:\n${JSON.stringify(report)}` };
  if (state.video.frames.length) userMessage.images = state.video.frames.map((frame) => frame.base64);
  try {
    const body = await callGemmaEndpoint({
      model: el.gemmaModelInput.value.trim() || DEFAULT_MODEL,
      local_model_path: el.localGemmaPathInput?.value.trim() || "",
      provider: getGemmaProviderPreference(),
      messages: [userMessage],
      stream: false,
      format: "json",
      options: { temperature: 0.2, num_predict: 420 }
    });
    state.gemma = { status: "Generated by local Gemma 4", output: normalizeGemmaOutput(body.content), error: null, running: false };
  } catch (error) {
    state.gemma = { status: "Local Gemma runtime unavailable", output: null, error: error.message, running: false };
  }
  renderGemma();
}

function renderGemma() {
  el.gemmaStatus.textContent = state.gemma.status;
  el.generateGemmaButton.disabled = state.gemma.running;
  el.generateGemmaButton.textContent = state.gemma.running ? "Generating" : "Generate";
  if (state.gemma.output) {
    const output = state.gemma.output;
    el.gemmaOutput.innerHTML = `<article class="gemma-card"><dl><div><dt>Cue</dt><dd>${escapeHtml(output.correction_cue || "")}</dd></div><div><dt>Drill</dt><dd>${escapeHtml(output.drill || "")}</dd></div><div><dt>Next run</dt><dd>${escapeHtml(output.next_run_focus || "")}</dd></div><div><dt>Video</dt><dd>${escapeHtml(output.visual_observations || "")}</dd></div><div><dt>Research</dt><dd>${escapeHtml(output.research_notes || "")}</dd></div></dl></article>`;
  } else if (state.gemma.error) el.gemmaOutput.innerHTML = `<p class="gemma-error">${escapeHtml(state.gemma.error)}</p>`;
  else el.gemmaOutput.innerHTML = "";
}

async function submitChat(e) {
  if (e) e.preventDefault();
  const text = el.chatInput?.value?.trim();
  if (!text || state.chat.running) return;
  
  if (el.chatInput) el.chatInput.value = "";
  addChatMessage("user", text);
  
  if (el.chatWelcome) el.chatWelcome.style.display = "none";
  state.chat.running = true;
  
  const loaderId = addChatBubble("coach", "Connecting to your Gemma 4 Coach...", true);
  
  try {
    const currentScore = state.vision.liveLatest?.assessment?.score || null;
    const activeIssues = state.vision.liveLatest?.assessment?.issues?.map(i => i.label) || [];
    
    const systemPrompt = `You are FormForward, an advanced AI Biomechanics and running form coach. 
We are currently in an interactive, multi-turn coaching chat. 

RUNNER BIOMETRIC DATA FOR YOUR AWARENESS:
- Current biomechanical form score: ${currentScore ? currentScore + "/100" : "Not measured yet (tell them to record dynamic video/live tracking)"}
- Logged dynamic mistakes: ${activeIssues.length ? activeIssues.join(", ") : "None critical measured"}
- Metric Summary Payload: ${JSON.stringify(state.analysis?.gemmaPayload || {})}
- Scraped Biomechanics Research: ${JSON.stringify(state.research.sources.filter(s => s.ok).slice(0, 2).map(s => ({ title: s.title, context: s.summary })))}

CONVERSATION INSTRUCTIONS:
1. Maintain a dynamic, encouraging, high-energy, yet professional coaching presence.
2. Use the biomechanical context above to offer DIRECT, CUSTOM solutions. NEVER make up numbers. Reference their exact mistakes (like overstriding or torso lean angle).
3. Do NOT output raw JSON. Talk back naturally in standard chat text. 
4. Give quick, action-oriented advice. Keep paragraphs compact and readable. 
5. Frame suggestions safely as coaching insights, not medical diagnoses.`;

    const messages = [
      { role: "system", content: systemPrompt },
      ...state.chat.history.map(h => ({ role: h.role === "user" ? "user" : "assistant", content: h.content })),
      { role: "user", content: text }
    ];

    const body = await callGemmaEndpoint({
      model: el.gemmaModelInput.value.trim() || DEFAULT_MODEL,
      local_model_path: el.localGemmaPathInput?.value.trim() || "",
      provider: getGemmaProviderPreference(),
      messages: messages,
      stream: false,
      format: "text",
      options: { temperature: 0.4, num_predict: 350 }
    });
    
    const responseText = body.content || "FormForward Coach successfully processed your telemetry.";
    
    const bubbleNode = document.getElementById(loaderId);
    if (bubbleNode) {
      const container = bubbleNode.querySelector(".chat-content");
      if (container) {
        container.textContent = responseText;
        container.classList.remove("typing-dots");
        container.style.color = "var(--cream)";
      }
      bubbleNode.removeAttribute("id");
    }
    
    state.chat.history.push({ role: "user", content: text });
    state.chat.history.push({ role: "coach", content: responseText });
    
    // Gamification reward
    if (typeof gainXp === "function") {
      gainXp(5);
    }
    
  } catch (err) {
    console.warn("Coach query failed:", err);
    const bubbleNode = document.getElementById(loaderId);
    if (bubbleNode) {
      const container = bubbleNode.querySelector(".chat-content");
      if (container) {
        container.textContent = "⚡ FormForward is waiting for local Gemma runtime: " + err.message;
        container.classList.remove("typing-dots");
        container.style.color = "var(--red)";
      }
    }
  } finally {
    state.chat.running = false;
    scrollToBottom();
  }
}

function addChatMessage(role, text) {
  addChatBubble(role, text);
  scrollToBottom();
}

function addChatBubble(role, text, isLoader = false) {
  if (!el.chatArea) return null;
  const bubbleId = `cb-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
  const row = document.createElement("div");
  row.id = bubbleId;
  row.style.display = "flex";
  row.style.gap = "12px";
  row.style.maxWidth = "88%";
  row.style.alignSelf = role === "user" ? "flex-end" : "flex-start";
  row.style.flexDirection = role === "user" ? "row-reverse" : "row";
  row.style.animation = "slideBubbleIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards";
  
  const icon = document.createElement("div");
  icon.style.width = "34px";
  icon.style.height = "34px";
  icon.style.borderRadius = "50%";
  icon.style.display = "grid";
  icon.style.placeItems = "center";
  icon.style.flexShrink = "0";
  icon.style.fontSize = "1rem";
  
  if (role === "user") {
    icon.textContent = "🏃";
    icon.style.background = "rgba(255,255,255,0.1)";
    icon.style.border = "2px solid #fff";
  } else {
    icon.textContent = "🤖";
    icon.style.background = "linear-gradient(135deg, var(--accent), var(--violet))";
    icon.style.border = "2px solid #fff";
    icon.style.boxShadow = "0 0 10px rgba(0, 240, 255, 0.2)";
  }
  
  const box = document.createElement("div");
  box.className = "chat-content";
  box.style.padding = "12px 16px";
  box.style.borderRadius = role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px";
  box.style.fontSize = "0.92rem";
  box.style.lineHeight = "1.5";
  box.style.whiteSpace = "pre-wrap";
  box.style.wordBreak = "break-word";
  
  if (role === "user") {
    box.style.background = "rgba(0, 240, 255, 0.15)";
    box.style.border = "2px solid var(--accent)";
    box.style.color = "#fff";
  } else {
    box.style.background = "var(--surface)";
    box.style.border = "2px solid rgba(255, 255, 255, 0.1)";
    box.style.color = "var(--muted)";
  }
  
  box.textContent = text;
  if (isLoader) {
    box.classList.add("typing-dots");
  }
  
  row.appendChild(icon);
  row.appendChild(box);
  el.chatArea.appendChild(row);
  
  return bubbleId;
}

function resetChat() {
  if (!el.chatArea) return;
  state.chat.history = [];
  
  // Keep just the welcome screen
  el.chatArea.innerHTML = "";
  if (el.chatWelcome) {
    el.chatArea.appendChild(el.chatWelcome);
    el.chatWelcome.style.display = "block";
  }
}

function scrollToBottom() {
  if (el.chatArea) {
    el.chatArea.scrollTo({ top: el.chatArea.scrollHeight, behavior: "smooth" });
  }
}

function normalizeGemmaOutput(raw) {
  let parsed = {};
  try {
    const text = String(raw).trim();
    const jsonText = text.startsWith("{") ? text : text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
    parsed = JSON.parse(jsonText);
  } catch {
    parsed = {};
  }
  const fallback = state.analysis.patterns[0] || {};
  return {
    correction_cue: cleanGemmaField(parsed.correction_cue || parsed.cue || parsed.correction || parsed.correctionCue) || fallback.cue || "",
    drill: cleanGemmaField(parsed.drill || parsed.drill_prescription || parsed.exercise) || fallback.drill || "",
    next_run_focus: cleanGemmaField(parsed.next_run_focus || parsed.nextRunFocus || parsed.focus || parsed.next_run) || fallback.nextRunFocus || "",
    visual_observations: cleanGemmaField(parsed.visual_observations || parsed.visualObservations || parsed.video || parsed.video_observations),
    research_notes: cleanGemmaField(parsed.research_notes || parsed.researchNotes || parsed.research || parsed.source_notes)
  };
}

function safeJson(raw) {
  try {
    const text = String(raw).trim();
    const jsonText = text.startsWith("{") ? text : text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
    return JSON.parse(jsonText);
  } catch {
    return {};
  }
}

function cleanGemmaField(value) {
  if (typeof value === "string") return value.trim();
  if (Array.isArray(value)) return value.map(cleanGemmaField).filter(Boolean).join(" ");
  if (value && typeof value === "object") return Object.values(value).map(cleanGemmaField).filter(Boolean).join(" ");
  return "";
}

function renderPayload() {
  el.payloadOutput.textContent = JSON.stringify(buildReport(), null, 2);
}

function buildReport() {
  const visionData = buildVisionPayload(state.vision.frameAnalyses);
  return {
    ...state.analysis.gemmaPayload,
    video_context: { file_name: state.video.name || null, sampled_frames: state.video.frames.length },
    cv_analysis: visionData,
    location_context: {
      status: state.gps.status,
      fix_count: state.gps.points.length,
      distance_km: round(state.gps.distanceMeters / 1000, 2),
      average_pace_min_per_km: averageGpsPace(state.gps),
      last_fix: state.gps.lastFix ? {
        lat: state.gps.lastFix.lat,
        lon: state.gps.lastFix.lon,
        accuracy_m: state.gps.lastFix.accuracy,
        timestamp: new Date(state.gps.lastFix.timestamp).toISOString()
      } : null
    },
    research_context: state.research.sources.filter((source) => source.ok).map(({ title, url, summary }) => ({ title, url, summary }))
  };
}

function renderCoverage() {
  el.availableMetrics.innerHTML = state.analysis.availableMetrics.map((key) => `<span class="tag">${METRICS[key].label}</span>`).join("");
  el.missingMetrics.innerHTML = state.analysis.missingMetrics.length ? state.analysis.missingMetrics.map((key) => `<span class="tag">${METRICS[key].label}</span>`).join("") : `<span class="tag">None</span>`;
}

function appendSvg(tagName, attrs, textContent) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tagName);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (textContent != null) node.textContent = textContent;
  el.metricChart.appendChild(node);
}

function buildSampleGpx() {
  return `<?xml version="1.0"?><gpx><trk><trkseg>${[178, 178, 179, 177, 176, 174, 172, 169, 166, 164, 162, 160].map((cadence, index) => `<trkpt lat="${(1.3 + index * 0.00168).toFixed(6)}" lon="103.8"><ele>${22 + index}</ele><time>${new Date(Date.UTC(2026, 4, 4, 0, index, 0)).toISOString()}</time><extensions><gpxtpx:hr>${142 + index * 3}</gpxtpx:hr><gpxtpx:cad>${Math.round(cadence / 2)}</gpxtpx:cad><ff:VerticalOscillation>${78 + index * 2}</ff:VerticalOscillation><ff:GroundContactTime>${232 + index * 5}</ff:GroundContactTime><ff:StrideLength>${(0.95 + index * 0.009).toFixed(2)}</ff:StrideLength><ff:GroundContactBalance>${(50.2 + index * 0.55).toFixed(1)}</ff:GroundContactBalance></extensions></trkpt>`).join("")}</trkseg></trk></gpx>`;
}

function tag(xml, name) {
  return xml.match(new RegExp(`<(?:(?:[\\w-]+):)?${name}\\b[^>]*>([\\s\\S]*?)<\\/(?:(?:[\\w-]+):)?${name}>`, "i"))?.[1]?.replace(/<[^>]+>/g, "").trim() || "";
}
function averageGpsPace(gpsState) {
  if (!gpsState.startedAt || gpsState.distanceMeters < 25) return null;
  const elapsedMinutes = (Date.now() - gpsState.startedAt) / 60000;
  const distanceKm = gpsState.distanceMeters / 1000;
  return distanceKm > 0 ? round(elapsedMinutes / distanceKm, 2) : null;
}
function haversineMeters(a, b) {
  const toRad = (value) => value * Math.PI / 180;
  const earthRadius = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * earthRadius * Math.asin(Math.sqrt(h));
}
function firstNumber(xml, names) {
  for (const name of names) {
    const value = number(tag(xml, name));
    if (Number.isFinite(value)) return value;
  }
  return null;
}
function parseTime(value, firstDate) {
  if (!value) return { seconds: null };
  if (/^\d+(\.\d+)?$/.test(value)) return { seconds: Number(value) };
  if (value.includes(":")) {
    const parts = value.split(":").map(Number);
    return { seconds: parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2] };
  }
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? { seconds: null } : { seconds: firstDate ? (date - firstDate) / 1000 : 0, date };
}
function parsePace(value) {
  if (!value) return null;
  if (String(value).includes(":")) {
    const [minutes, seconds] = String(value).split(":").map(Number);
    return minutes + seconds / 60;
  }
  return number(value);
}
function number(value) {
  const match = String(value ?? "").match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : null;
}
function normalizeCadence(value) {
  return Number.isFinite(value) && value < 120 ? value * 2 : value;
}
function avg(values) {
  const valid = values.filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}
function pct(value, base) {
  return Number.isFinite(value) && Number.isFinite(base) && base ? ((value - base) / base) * 100 : null;
}
function collectEvidence(events, types) {
  return events.filter((event) => types.includes(event.type)).flatMap((event) => event.signals).slice(0, 4);
}
function estimateDistance(samples) {
  let km = 0;
  for (let index = 1; index < samples.length; index += 1) {
    const delta = samples[index].timeSeconds - samples[index - 1].timeSeconds;
    if (Number.isFinite(samples[index].pace) && samples[index].pace > 0) km += delta / (samples[index].pace * 60);
  }
  return km;
}
function formatDuration(seconds) {
  const rounded = Math.max(0, Math.round(seconds || 0));
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
}
function formatMetric(value, key) {
  if (!Number.isFinite(value)) return "n/a";
  if (key === "pace") return `${Math.floor(value)}:${String(Math.round((value % 1) * 60)).padStart(2, "0")}`;
  if (key === "strideLength") return value.toFixed(2);
  return String(Math.round(value));
}
function round(value, places = 0) {
  if (!Number.isFinite(value)) return null;
  return Math.round(value * 10 ** places) / 10 ** places;
}
function once(target, eventName) {
  return new Promise((resolve, reject) => {
    target.addEventListener(eventName, resolve, { once: true });
    target.addEventListener("error", () => reject(new Error(`${eventName} failed`)), { once: true });
  });
}
function describeCameraError(err) {
  if (!err) return "Camera access failed for an unknown reason.";
  const message = typeof err.message === "string" ? err.message : "";
  switch (err.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera access was blocked. Allow camera permission for localhost in the browser. If you are using the in-app browser and it still stays blocked, open the same localhost URL in Chrome or Safari and retry live tracking there.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found on this device. Connect a camera or switch to a device with one available.";
    case "NotReadableError":
    case "TrackStartError":
      return "Your camera is busy in another app. Close the other app using the camera, then retry live tracking.";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "The requested camera settings were not available. Retry live tracking and the app will use the closest supported camera mode.";
    case "AbortError":
      return "The browser interrupted camera startup. Try starting live tracking one more time.";
    default:
      return message || "Camera access failed. Check browser permissions and try again.";
  }
}
function compactUrl(url) {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname === "/" ? "" : parsed.pathname}`;
  } catch {
    return url;
  }
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[char]);
}
function qs(selector) {
  return document.querySelector(selector);
}

// ═══════════ WhatsApp Video Gallery ═══════════
const whatsappVideoGrid = qs("#whatsappVideoGrid");
const whatsappStatus = qs("#whatsappStatus");
const loadWhatsappButton = qs("#loadWhatsappButton");

if (loadWhatsappButton) {
  loadWhatsappButton.addEventListener("click", loadWhatsAppVideos);
}

async function loadWhatsAppVideos() {
  if (whatsappStatus) whatsappStatus.textContent = "Loading WhatsApp videos...";
  try {
    const response = await fetch(getApiUrl("/api/whatsapp-videos"));
    const data = await response.json();
    if (!data.videos?.length) {
      whatsappStatus.textContent = "No WhatsApp videos found in data/videos/.";
      return;
    }
    whatsappVideoGrid.innerHTML = data.videos.map((v, i) => `
      <div class="whatsapp-video-card" style="background: rgba(37,211,102,0.05); border: 1px solid rgba(37,211,102,0.2); border-radius: 12px; overflow: hidden; transition: all 0.3s ease;">
        <video src="${escapeHtml(getApiUrl(v.path))}" preload="metadata" muted playsinline style="width: 100%; aspect-ratio: 16/9; object-fit: cover; background: #000; cursor: pointer;"
          onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;"></video>
        <div style="padding: 10px;">
          <strong style="font-size: 0.85rem; color: #fff;">Run Video ${i + 1}</strong>
          <p style="font-size: 0.75rem; color: var(--muted); margin-top: 2px;">${escapeHtml(v.name)} • ${v.sizeMB} MB</p>
          <div style="display: flex; gap: 6px; margin-top: 8px;">
            <button type="button" class="wa-analyze-btn" data-path="${escapeHtml(v.path)}" style="flex: 1; background: linear-gradient(135deg, #25d366, #128c7e); color: #fff; border: none; padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; cursor: pointer;">🔍 Analyze Pose</button>
            <button type="button" class="wa-train-btn" data-path="${escapeHtml(v.path)}" style="flex: 1; background: rgba(243,161,43,0.2); color: var(--amber); border: 1px solid rgba(243,161,43,0.3); padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; cursor: pointer;">⚡ Train Gemma</button>
          </div>
        </div>
      </div>
    `).join("");

    // Bind analyze buttons
    whatsappVideoGrid.querySelectorAll(".wa-analyze-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        const videoPath = btn.dataset.path;
        btn.textContent = "⏳ Analyzing…";
        btn.disabled = true;
        try {
          const resp = await fetch(getApiUrl(videoPath));
          const blob = await resp.blob();
          const file = new File([blob], videoPath.split("/").pop(), { type: "video/mp4" });
          await handleVideoUpload(file);
          switchTab("video");
          setTimeout(() => {
            runVisionAnalysis();
            el.visionPanel.scrollIntoView({ behavior: "smooth" });
          }, 800);
          unlockAchievement("badge-video");
          gainXp(30);
        } catch (e) {
          btn.textContent = "❌ Failed";
        }
      });
    });

    // Bind train buttons
    whatsappVideoGrid.querySelectorAll(".wa-train-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        const videoPath = btn.dataset.path;
        btn.textContent = "⏳ Training…";
        btn.disabled = true;
        try {
          const resp = await fetch(getApiUrl(videoPath));
          const blob = await resp.blob();
          const file = new File([blob], videoPath.split("/").pop(), { type: "video/mp4" });
          await handleVideoUpload(file);
          await runVisionAnalysis();
          await runFullPipeline();
          if (state.gemma.error) throw new Error(state.gemma.error);
          btn.textContent = "✅ Trained";
          gainXp(40);
          switchTab("coach");
        } catch (e) {
          btn.textContent = "❌ Failed";
        }
      });
    });

    whatsappStatus.textContent = `${data.videos.length} WhatsApp videos loaded. Hover to preview, click to analyze.`;
    gainXp(15);
  } catch (err) {
    whatsappStatus.textContent = `Failed to load videos: ${err.message}`;
  }
}

// ═══════════ FIT Data Library ═══════════
const fitFileGrid = qs("#fitFileGrid");
const fitTrainingStatus = qs("#fitTrainingStatus");
const loadFitLibraryButton = qs("#loadFitLibraryButton");
const trainAllFitButton = qs("#trainAllFitButton");

if (loadFitLibraryButton) loadFitLibraryButton.addEventListener("click", loadFitLibrary);
if (trainAllFitButton) trainAllFitButton.addEventListener("click", trainAllFitData);

const fitLibrary = { files: [], parsedRuns: [], loaded: false };

async function loadFitLibrary() {
  if (fitTrainingStatus) fitTrainingStatus.textContent = "Scanning FIT library…";
  try {
    const response = await fetch(getApiUrl("/api/fit-files"));
    const data = await response.json();
    fitLibrary.files = data.files || [];
    if (!fitLibrary.files.length) {
      fitTrainingStatus.textContent = "No FIT files found in data/fit/.";
      return;
    }
    fitFileGrid.innerHTML = fitLibrary.files.map((f, i) => `
      <div class="fit-file-card" data-index="${i}" style="background: rgba(243,161,43,0.06); border: 1px solid rgba(243,161,43,0.15); border-radius: 8px; padding: 10px; cursor: pointer; transition: all 0.2s ease; display: flex; flex-direction: column; gap: 4px;"
        onmouseenter="this.style.borderColor='rgba(243,161,43,0.5)'; this.style.background='rgba(243,161,43,0.12)';"
        onmouseleave="this.style.borderColor='rgba(243,161,43,0.15)'; this.style.background='rgba(243,161,43,0.06)';">
        <strong style="font-size: 0.78rem; color: var(--amber); word-break: break-all;">📊 ${escapeHtml(f.name.replace("_ACTIVITY.fit",""))}</strong>
        <span style="font-size: 0.7rem; color: var(--muted);">${f.sizeKB} KB</span>
      </div>
    `).join("");

    // Click to load individual FIT file
    fitFileGrid.querySelectorAll(".fit-file-card").forEach(card => {
      card.addEventListener("click", async () => {
        const idx = parseInt(card.dataset.index);
        const f = fitLibrary.files[idx];
        card.style.opacity = "0.5";
        try {
          const resp = await fetch(getApiUrl(f.path));
          const buffer = await resp.arrayBuffer();
          loadFitActivity(buffer, f.name);
          card.style.opacity = "1";
          card.style.borderColor = "#7ed6af";
          fitTrainingStatus.textContent = `Loaded ${f.name}`;
        } catch (e) {
          card.style.opacity = "1";
          fitTrainingStatus.textContent = `Failed to load ${f.name}: ${e.message}`;
        }
      });
    });

    fitLibrary.loaded = true;
    fitTrainingStatus.textContent = `${fitLibrary.files.length} FIT files ready. Click any to load, or hit "Train Gemma on All" for bulk training.`;
    gainXp(20);
  } catch (err) {
    fitTrainingStatus.textContent = `Failed: ${err.message}`;
  }
}

async function trainAllFitData() {
  if (!fitLibrary.files.length) {
    await loadFitLibrary();
  }
  if (!fitLibrary.files.length) return;

  const total = fitLibrary.files.length;
  fitTrainingStatus.textContent = `Parsing ${total} FIT files for Gemma 4 training…`;
  fitLibrary.parsedRuns = [];

  let processed = 0;
  let errors = 0;
  const batchSize = 5; // parse in batches to not block UI

  for (let i = 0; i < total; i += batchSize) {
    const batch = fitLibrary.files.slice(i, i + batchSize);
    await Promise.all(batch.map(async (f) => {
      try {
        const resp = await fetch(getApiUrl(f.path));
        const buffer = await resp.arrayBuffer();
        const result = parseFitFile(buffer);
        if (result.rows.length > 0) {
          fitLibrary.parsedRuns.push({
            name: f.name,
            summary: result.summary,
            rows: result.rows.filter((_, j) => j % 10 === 0) // Downsample for training context
          });
        }
        processed++;
      } catch {
        errors++;
        processed++;
      }
      fitTrainingStatus.textContent = `Parsed ${processed}/${total} FIT files (${errors} errors)…`;
    }));
  }

  fitTrainingStatus.textContent = `✅ Parsed ${fitLibrary.parsedRuns.length} valid runs from ${total} files. Sending to Gemma 4…`;

  // Build training context from all parsed FIT data
  const trainingContext = {
    total_runs: fitLibrary.parsedRuns.length,
    summaries: fitLibrary.parsedRuns.map(r => ({
      file: r.name,
      duration_min: r.summary.durationMinutes,
      avg_cadence: r.summary.avgCadence,
      avg_hr: r.summary.avgHR,
      avg_pace: r.summary.avgPace,
      records: r.summary.totalRecords
    })),
    // Include detailed data from first 5 runs as exemplars
    exemplar_runs: fitLibrary.parsedRuns.slice(0, 5).map(r => ({
      file: r.name,
      data_points: r.rows.slice(0, 30)
    }))
  };

  // Feed into the Gemma pipeline
  try {
    const response = await fetch(getApiUrl("/api/analyze-form"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_analysis: {
          ...state.analysis?.gemmaPayload,
          fit_training_data: trainingContext
        },
        research_sources: state.research.sources,
        cv_analysis: buildVisionPayload(state.vision.frameAnalyses),
        model: el.gemmaModelInput.value.trim() || DEFAULT_MODEL,
        provider: getGemmaProviderPreference(),
        local_model_path: el.localGemmaPathInput?.value.trim() || ""
      })
    });
    const result = await response.json();
    if (result.coaching) {
      renderOptimalFormOutput(result.coaching, result);
      state.gemma = {
        status: `Trained on ${fitLibrary.parsedRuns.length} FIT activities via ${result.model_used || "Gemma 4"}`,
        output: normalizeGemmaOutput(JSON.stringify(result.coaching)),
        error: null,
        running: false
      };
      renderGemma();
      renderPayload();
      unlockAchievement("badge-gemma");
      gainXp(100);
    }
    fitTrainingStatus.textContent = `✅ Gemma 4 trained on ${fitLibrary.parsedRuns.length} runs. Form adjustments updated on Dashboard.`;
  } catch (err) {
    fitTrainingStatus.textContent = `❌ Training failed: ${err.message}. FIT data parsed successfully — retry when Gemma is online.`;
  }
}

// ═══════════ Calibration & Inference Engine ═══════════

/**
 * Run the full calibration pipeline:
 * 1. Extract frames from calibration video
 * 2. Run pose detection on each frame
 * 3. Compute calibration heuristics (torso lean, cadence, asymmetry, bounce)
 * 4. Parse the matching FIT/CSV telemetry
 * 5. Synchronize video frames with telemetry by timestamp
 * 6. Build a calibration profile (regression + kNN lookup table)
 */
async function runCalibration() {
  const videoFile = state.calibration.calibVideoFile;
  const fitFile = state.calibration.calibFitFile;
  if (!videoFile) { window.alert("Please select a calibration video."); return; }
  if (!fitFile) { window.alert("Please select a matching FIT/CSV file."); return; }

  const statusEl = el.calibrationStatus;
  statusEl.textContent = "Extracting video frames...";
  el.runCalibrationButton.disabled = true;

  try {
    // Step 1: Extract frames from calibration video (denser sampling for calibration)
    const videoUrl = URL.createObjectURL(videoFile);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.src = videoUrl;
    await once(video, "loadedmetadata");

    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 720 / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    const fps = 30; // assumed
    const durationMs = video.duration * 1000;

    // Sample more frames for calibration (every 200ms or ~5 fps)
    const sampleIntervalMs = 200;
    const frameCount = Math.min(50, Math.floor(durationMs / sampleIntervalMs));
    const calibFrames = [];

    statusEl.textContent = `Extracting ${frameCount} frames from calibration video...`;

    for (let i = 0; i < frameCount; i++) {
      const timeMs = i * sampleIntervalMs;
      video.currentTime = Math.min(video.duration - 0.05, timeMs / 1000);
      await once(video, "seeked");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.74);
      calibFrames.push({ label: `${Math.round(timeMs)}ms`, dataUrl, timestampMs: timeMs });
    }
    URL.revokeObjectURL(videoUrl);

    // Step 2: Run pose detection on each frame
    statusEl.textContent = `Running pose detection on ${calibFrames.length} frames...`;
    const frameAnalyses = [];
    for (const frame of calibFrames) {
      if (visionReady()) {
        const img = new Image();
        img.src = frame.dataUrl;
        await new Promise(r => { img.onload = r; });
        const landmarks = detectPose(img);
        const angles = landmarks ? computeAngles(landmarks) : null;
        const assessment = angles ? assessForm(angles) : null;
        frameAnalyses.push({ label: frame.label, landmarks, angles, assessment, timestampMs: frame.timestampMs });
      } else {
        frameAnalyses.push({ label: frame.label, landmarks: null, angles: null, assessment: null, timestampMs: frame.timestampMs });
      }
    }

    const detectedCount = frameAnalyses.filter(f => f.landmarks).length;
    statusEl.textContent = `Pose detected in ${detectedCount}/${calibFrames.length} frames. Computing heuristics...`;

    // Step 3: Compute calibration heuristics
    const heuristics = computeCalibrationHeuristics(frameAnalyses, fps);
    state.calibration.heuristics = heuristics;

    // Step 4: Parse the FIT/CSV telemetry
    statusEl.textContent = "Parsing telemetry data...";
    let telemetryRows;
    if (fitFile.name.toLowerCase().endsWith(".fit")) {
      const buffer = await fitFile.arrayBuffer();
      const { rows } = parseFitFile(buffer);
      telemetryRows = rows.map((r, i) => ({
        timeSeconds: i,
        cadence: r.cadence,
        verticalOscillation: r.vertical_oscillation_mm,
        groundContactTime: r.ground_contact_time_ms,
        strideLength: r.stride_length_m,
        pace: r.pace,
        heartRate: r.heart_rate,
        gctBalance: r.gct_balance
      }));
    } else {
      const text = await fitFile.text();
      telemetryRows = parseInferenceCsv(text);
    }

    // Step 5: Synchronize
    statusEl.textContent = "Synchronizing video and telemetry...";
    const videoOffsetMin = parseFloat(el.calibVideoOffsetInput?.value || "0");
    const syncResult = synchronize({
      videoHeuristics: heuristics.perFrame,
      telemetryRows,
      videoStartOffsetSeconds: videoOffsetMin * 60,
      maxGapSeconds: 5
    });
    state.calibration.syncedPairs = syncResult.pairs;

    // Step 6: Build calibration profile
    statusEl.textContent = "Building calibration profile...";
    const profile = buildCalibrationProfile(syncResult.pairs);
    state.calibration.profile = profile;

    if (!profile.valid) {
      statusEl.textContent = `Calibration failed: ${profile.reason}`;
      el.runCalibrationButton.disabled = false;
      return;
    }

    // Render profile summary
    renderCalibrationProfile(profile, heuristics, syncResult.stats);
    statusEl.textContent = `Calibration complete. ${syncResult.stats.matched} pairs synchronized (avg gap: ${syncResult.stats.avgGapMs}ms). Profile ready for inference.`;
    state.calibration.status = statusEl.textContent;

    // Enable inference
    el.inferenceStatus.textContent = "Calibration profile loaded. Upload a FIT file to start inference.";
    if (state.inference.telemetryRows.length > 0) {
      el.startInferenceButton.disabled = false;
    }

  } catch (err) {
    statusEl.textContent = `Calibration error: ${err.message}`;
    console.error("Calibration failed:", err);
  }

  el.runCalibrationButton.disabled = false;
}

function renderCalibrationProfile(profile, heuristics, syncStats) {
  if (!el.calibProfilePanel) return;
  el.calibProfilePanel.style.display = "block";

  // Summary
  el.calibProfileSummary.innerHTML = `
    <p><strong>Samples:</strong> ${profile.sampleCount} synchronized pairs</p>
    <p><strong>Avg Sync Gap:</strong> ${syncStats.avgGapMs}ms</p>
    <p><strong>Video Cadence:</strong> ${heuristics.cadence ?? "N/A"} spm</p>
    <p><strong>Avg Torso Lean:</strong> ${heuristics.torsoLean ?? "N/A"}&deg;</p>
    <p><strong>Avg Vertical Bounce:</strong> ${heuristics.verticalBounce ?? "N/A"}</p>
    <p><strong>Avg Stride Asymmetry:</strong> ${heuristics.strideAsymmetry ?? "N/A"}</p>
  `;

  // Regression cards
  const grid = el.calibRegressionGrid;
  grid.innerHTML = "";
  for (const poseKey of profile.poseKeys) {
    for (const telKey of profile.telemetryKeys) {
      const reg = profile.regressions[poseKey]?.[telKey];
      if (!reg || reg.r2 === null || reg.r2 < 0.01) continue;
      const r2Pct = Math.round(reg.r2 * 100);
      const color = r2Pct > 50 ? "#7ed6af" : r2Pct > 25 ? "#f3a12b" : "var(--muted)";
      const card = document.createElement("div");
      card.style.cssText = `background: rgba(255,255,255,0.03); border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-size: 0.8rem;`;
      card.innerHTML = `
        <div style="color: ${color}; font-weight: 700; margin-bottom: 4px;">${telKey} &rarr; ${poseKey}</div>
        <div style="color: var(--muted);">R&sup2; = ${r2Pct}% (n=${reg.n})</div>
        <div style="color: var(--muted);">y = ${reg.slope}x + ${reg.intercept}</div>
      `;
      grid.appendChild(card);
    }
  }
}

/**
 * Parse a CSV file into telemetry rows for inference.
 */
function parseInferenceCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = lines.shift().split(",").map(h => h.trim().toLowerCase().replace(/[^a-z0-9]/g, ""));
  return lines.map((line, i) => {
    const values = line.split(",").map(v => v.trim());
    const row = { timeSeconds: i };
    headers.forEach((h, j) => {
      const val = parseFloat(values[j]);
      if (h.includes("cadence")) row.cadence = val;
      else if (h.includes("verticaloscillation") || h === "vo") row.verticalOscillation = val;
      else if (h.includes("groundcontacttime") || h === "gct") row.groundContactTime = val;
      else if (h.includes("stridelength") || h === "stride") row.strideLength = val;
      else if (h.includes("pace")) row.pace = val;
      else if (h.includes("heartrate") || h === "hr") row.heartRate = val;
      else if (h.includes("gctbalance")) row.gctBalance = val;
    });
    return row;
  });
}

/**
 * Start the real-time inference demo with a simulated Garmin stream.
 */
function startInferenceDemo() {
  if (!state.calibration.profile?.valid) {
    el.inferenceStatus.textContent = "No valid calibration profile. Run calibration first.";
    return;
  }
  if (!state.inference.telemetryRows.length) {
    el.inferenceStatus.textContent = "No telemetry data loaded. Upload a FIT/CSV file.";
    return;
  }

  // Stop any existing stream
  stopInferenceDemo();

  const speed = parseInt(el.inferenceSpeedSelect?.value || "10");
  el.inferenceDashboard.style.display = "block";
  el.startInferenceButton.disabled = true;
  el.stopInferenceButton.disabled = false;
  state.inference.running = true;

  // Create inference engine
  state.inference.engine = createInferenceEngine(state.calibration.profile, {
    windowSize: 5,
    cueIntervalSeconds: 15,
    onFormUpdate({ inferred, changes, confidence }) {
      // Update dashboard cards
      el.inferTorsoLean.textContent = inferred.torsoLean != null ? `${inferred.torsoLean}\u00B0` : "--";
      el.inferVerticalBounce.textContent = inferred.verticalBounce != null ? `${inferred.verticalBounce}` : "--";
      el.inferStrideAsymmetry.textContent = inferred.strideAsymmetry != null ? `${inferred.strideAsymmetry}` : "--";
      el.inferConfidence.textContent = confidence || "--";

      // Color code confidence
      const confColors = { high: "#7ed6af", medium: "#f3a12b", low: "#f36b6d" };
      el.inferConfidence.style.color = confColors[confidence] || "var(--text)";

      // Render form changes
      if (changes.length) {
        el.inferChangesList.innerHTML = changes.map(c => {
          const icon = c.severity === "bad" ? "\ud83d\udd34" : c.severity === "warn" ? "\ud83d\udfe1" : "\ud83d\udfe2";
          return `<div style="margin-bottom: 6px;">${icon} <strong>${c.metric}:</strong> ${c.current} (baseline: ${c.baseline}, ${c.direction} by ${Math.abs(c.delta)}) - ${c.description}</div>`;
        }).join("");
      } else {
        el.inferChangesList.innerHTML = `<div style="color: #7ed6af;">\u2713 All metrics within calibration baseline.</div>`;
      }
    },
    onCoachingCue({ cue, changes }) {
      state.inference.latestCue = cue;
      el.inferCoachingCues.innerHTML = `<div style="background: rgba(243,161,43,0.08); border: 1px solid rgba(243,161,43,0.2); border-radius: 8px; padding: 12px; margin-bottom: 8px; line-height: 1.5;">${escapeHtml(cue)}</div>` + el.inferCoachingCues.innerHTML;
    }
  });

  // Create simulated stream
  state.inference.stream = createSimulatedStream(state.inference.telemetryRows, {
    speedMultiplier: speed,
    onTick(row, index, total) {
      state.inference.engine.processSnapshot(row);
      const pct = Math.round((index / total) * 100);
      el.inferProgressBar.style.width = `${pct}%`;
      el.inferProgressLabel.textContent = `${index + 1} / ${total} (${pct}%)`;
    },
    onComplete() {
      state.inference.running = false;
      el.inferenceStatus.textContent = "Stream complete.";
      el.startInferenceButton.disabled = false;
      el.stopInferenceButton.disabled = true;
      el.inferProgressBar.style.width = "100%";
    }
  });

  el.inferenceStatus.textContent = `Streaming at ${speed}x speed...`;
  state.inference.stream.start();
}

function stopInferenceDemo() {
  if (state.inference.stream) {
    state.inference.stream.stop();
    state.inference.stream = null;
  }
  state.inference.running = false;
  el.startInferenceButton.disabled = !state.calibration.profile?.valid || !state.inference.telemetryRows.length;
  el.stopInferenceButton.disabled = true;
  el.inferenceStatus.textContent = state.calibration.profile?.valid
    ? "Stream stopped. Ready to restart."
    : "Load a calibration profile first.";
}

// ═══════════ OCR Dot Tracking ═══════════
const ocrDotsToggle = qs("#ocrDotsToggle");
const ocrDotStyle = qs("#ocrDotStyle");
const ocrOptimalCount = qs("#ocrOptimalCount");
const ocrWarnCount = qs("#ocrWarnCount");
const ocrCriticalCount = qs("#ocrCriticalCount");
const ocrTotalCount = qs("#ocrTotalCount");

let ocrDotsEnabled = false;
let ocrOverlayCanvas = null;

if (ocrDotsToggle) {
  ocrDotsToggle.addEventListener("change", () => {
    ocrDotsEnabled = ocrDotsToggle.checked;
    if (ocrDotsEnabled) {
      createOcrOverlay();
      if (state.vision.liveLatest?.landmarks) {
        renderOcrDots(state.vision.liveLatest.landmarks, state.vision.liveLatest.assessment);
      }
    } else {
      removeOcrOverlay();
    }
  });
}

function createOcrOverlay() {
  if (ocrOverlayCanvas) return;
  const video = el.videoPreview;
  if (!video) return;

  ocrOverlayCanvas = document.createElement("canvas");
  ocrOverlayCanvas.id = "ocrDotsOverlay";
  ocrOverlayCanvas.style.cssText = `
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 10;
  `;
  video.parentElement.style.position = "relative";
  video.parentElement.appendChild(ocrOverlayCanvas);
}

function removeOcrOverlay() {
  if (ocrOverlayCanvas) {
    ocrOverlayCanvas.remove();
    ocrOverlayCanvas = null;
  }
  if (ocrOptimalCount) ocrOptimalCount.textContent = "--";
  if (ocrWarnCount) ocrWarnCount.textContent = "--";
  if (ocrCriticalCount) ocrCriticalCount.textContent = "--";
  if (ocrTotalCount) ocrTotalCount.textContent = "--";
}

// MediaPipe landmark names for labeling OCR dots
const LANDMARK_NAMES = [
  "Nose", "L Eye In", "L Eye", "L Eye Out", "R Eye In", "R Eye", "R Eye Out",
  "L Ear", "R Ear", "Mouth L", "Mouth R", "L Shoulder", "R Shoulder",
  "L Elbow", "R Elbow", "L Wrist", "R Wrist", "L Pinky", "R Pinky",
  "L Index", "R Index", "L Thumb", "R Thumb", "L Hip", "R Hip",
  "L Knee", "R Knee", "L Ankle", "R Ankle", "L Heel", "R Heel",
  "L Foot Index", "R Foot Index"
];

// Key landmarks that map to running form assessment
const FORM_LANDMARKS = {
  11: "shoulder_l", 12: "shoulder_r", 13: "elbow_l", 14: "elbow_r",
  23: "hip_l", 24: "hip_r", 25: "knee_l", 26: "knee_r",
  27: "ankle_l", 28: "ankle_r", 0: "nose"
};

function renderOcrDots(landmarks, assessment) {
  if (!ocrDotsEnabled || !ocrOverlayCanvas || !landmarks) return;

  const video = el.videoPreview;
  const vw = video.videoWidth || video.clientWidth || 640;
  const vh = video.videoHeight || video.clientHeight || 480;
  ocrOverlayCanvas.width = vw;
  ocrOverlayCanvas.height = vh;
  const ctx = ocrOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, vw, vh);

  const style = ocrDotStyle?.value || "dots";
  const grades = assessment?.grades || {};
  let optimal = 0, warn = 0, critical = 0;

  // Grade each landmark based on the form assessment
  const landmarkGrades = {};
  landmarks.forEach((lm, i) => {
    const formKey = FORM_LANDMARKS[i];
    let grade = "good"; // default
    if (formKey) {
      // Map landmark to assessment keys
      if (formKey.includes("elbow") && (grades.leftElbow === "Bad" || grades.rightElbow === "Bad")) grade = "bad";
      else if (formKey.includes("elbow") && (grades.leftElbow === "Needs Improvement" || grades.rightElbow === "Needs Improvement")) grade = "warn";
      else if (formKey.includes("knee") && grades.kneeDrive === "Bad") grade = "bad";
      else if (formKey.includes("knee") && grades.kneeDrive === "Needs Improvement") grade = "warn";
      else if (formKey.includes("hip") && grades.hipDrop === "Bad") grade = "bad";
      else if (formKey.includes("hip") && grades.hipDrop === "Needs Improvement") grade = "warn";
      else if (formKey.includes("ankle") && (grades.leftHipAnkle === "Bad" || grades.rightHipAnkle === "Bad")) grade = "bad";
      else if (formKey.includes("ankle") && (grades.overstride === "Needs Improvement")) grade = "warn";
      else if (formKey === "nose" && grades.headAngle === "Bad") grade = "bad";
      else if (formKey === "nose" && grades.headAngle === "Needs Improvement") grade = "warn";
    }
    landmarkGrades[i] = grade;
    if (grade === "good") optimal++;
    else if (grade === "warn") warn++;
    else critical++;
  });

  const colors = { good: "#7ed6af", warn: "#f3a12b", bad: "#f36b6d" };

  if (style === "skeleton" || style === "heatmap") {
    // Draw skeleton connections
    const connections = [
      [11, 13], [13, 15], [12, 14], [14, 16], // arms
      [11, 12], [11, 23], [12, 24], [23, 24], // torso
      [23, 25], [25, 27], [24, 26], [26, 28], // legs
      [27, 29], [28, 30], [29, 31], [30, 32]  // feet
    ];
    ctx.lineWidth = 3;
    connections.forEach(([a, b]) => {
      if (landmarks[a] && landmarks[b]) {
        const gradeA = landmarkGrades[a] || "good";
        const gradeB = landmarkGrades[b] || "good";
        const worstGrade = gradeA === "bad" || gradeB === "bad" ? "bad" : gradeA === "warn" || gradeB === "warn" ? "warn" : "good";
        ctx.strokeStyle = colors[worstGrade];
        ctx.globalAlpha = style === "heatmap" ? 0.6 : 0.8;
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * vw, landmarks[a].y * vh);
        ctx.lineTo(landmarks[b].x * vw, landmarks[b].y * vh);
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }

  if (style === "heatmap") {
    // Draw heatmap circles around problematic areas
    landmarks.forEach((lm, i) => {
      const grade = landmarkGrades[i];
      if (grade !== "good" && lm.visibility > 0.3) {
        const radius = grade === "bad" ? 28 : 18;
        const gradient = ctx.createRadialGradient(lm.x * vw, lm.y * vh, 0, lm.x * vw, lm.y * vh, radius);
        gradient.addColorStop(0, grade === "bad" ? "rgba(243,107,109,0.5)" : "rgba(243,161,43,0.4)");
        gradient.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(lm.x * vw, lm.y * vh, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // Draw dots on all styles
  landmarks.forEach((lm, i) => {
    if (lm.visibility < 0.3) return;
    const grade = landmarkGrades[i];
    const x = lm.x * vw;
    const y = lm.y * vh;
    const radius = FORM_LANDMARKS[i] ? 7 : 4;

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
    ctx.fillStyle = colors[grade] + "40";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = colors[grade];
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label key landmarks
    if (FORM_LANDMARKS[i] && style !== "dots") {
      ctx.font = "bold 9px Inter, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText(LANDMARK_NAMES[i] || "", x, y - radius - 5);
    }
  });

  // Update stats
  if (ocrOptimalCount) ocrOptimalCount.textContent = optimal;
  if (ocrWarnCount) ocrWarnCount.textContent = warn;
  if (ocrCriticalCount) ocrCriticalCount.textContent = critical;
  if (ocrTotalCount) ocrTotalCount.textContent = landmarks.length;
}

// Hook into the live vision loop to render OCR dots
const _originalUpdateLive = updateLiveStateFromAnalysis;
updateLiveStateFromAnalysis = function patchedUpdateLive(analysis, opts) {
  _originalUpdateLive.call(this, analysis, opts);
  if (analysis?.landmarks && ocrDotsEnabled) {
    renderOcrDots(analysis.landmarks, analysis.assessment);
  }
};
// Monkey-patch — we call renderOcrDots from the live loop callback

// Auto-load WhatsApp videos and FIT library on start
setTimeout(() => {
  loadWhatsAppVideos();
  loadFitLibrary();
}, 1500);
