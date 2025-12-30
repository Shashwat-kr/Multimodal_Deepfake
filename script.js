// CONFIG
const API_BASE_URL = "http://localhost:5001/api";

let currentFile = null;
let analysisResults = null;

document.addEventListener("DOMContentLoaded", () => {
  drawLines();
  initUpload();
  initEventStream();
  initNotes();
  initThemeToggle();
});

// === THEME TOGGLE ===
function initThemeToggle() {
  const toggle = document.querySelector(".mode-toggle");
  if (!toggle) return;

  const saved = localStorage.getItem("mediashield-theme");
  if (saved === "light") {
    document.body.classList.add("theme-light");
  }

  toggle.addEventListener("click", () => {
    const isLight = document.body.classList.toggle("theme-light");
    localStorage.setItem("mediashield-theme", isLight ? "light" : "dark");
  });
}

// === BACKGROUND LINES ===
function drawLines() {
  const canvas = document.getElementById("bgLines");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
  }
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(250,204,21,0.22)";
    ctx.lineWidth = 0.7;
    const step = 80;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + step * 0.6, canvas.height);
      ctx.stroke();
    }
  }
  window.addEventListener("resize", resize);
  resize();
}

// === EVENT STREAM ===
function initEventStream() {
  const stream = document.getElementById("eventStream");
  if (!stream) return;
  const events = [
    "⏱ Models loaded (BERT, CLIP)",
    "🔍 SerpAPI connectivity available",
    "🧪 Sentiment & fake-news heads primed",
    "📡 Multimodal pipeline ready",
  ];
  events.forEach((e) => {
    const div = document.createElement("div");
    div.className = "mini-log-item";
    div.textContent = e;
    stream.appendChild(div);
  });
}

// === NOTES ===
function initNotes() {
  const list = document.getElementById("noteList");
  if (!list) return;
  const notes = [
    "Watch for extreme certainty alongside anonymous or vague sourcing.",
    "Visual-text mismatch plus low domain trust is often more important than sentiment alone.",
    "For election content, treat unverified viral clips as high-risk by default.",
  ];
  notes.forEach((t) => {
    const div = document.createElement("div");
    div.className = "note-item";
    div.textContent = t;
    list.appendChild(div);
  });
}

// === UPLOAD HANDLING ===
function initUpload() {
  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("fileInput");
  if (!uploadArea || !fileInput) return;

  uploadArea.addEventListener("click", () => fileInput.click());
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
  });
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
  });
}

function handleFileSelect(file) {
  const allowed = [
    "image/jpeg",
    "image/png",
    "image/jpg",
    "text/plain",
    "application/pdf",
  ];
  if (file.type && !allowed.includes(file.type)) {
    toast("Unsupported file type. Use JPG, PNG, TXT, or PDF.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    toast("File larger than 10MB is not allowed.", "error");
    return;
  }
  currentFile = file;
  showPreview(file);
}

function showPreview(file) {
  const preview = document.getElementById("filePreview");
  const upload = document.getElementById("uploadArea");
  const nameEl = document.getElementById("fileName");
  const sizeEl = document.getElementById("fileSize");
  const imgEl = document.getElementById("previewImage");

  upload.style.display = "block";
  preview.classList.remove("hidden");

  upload.style.display = "none";

  nameEl.textContent = file.name;
  sizeEl.textContent = formatSize(file.size) + " • " + (file.type || "unknown");
  imgEl.innerHTML = "";

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = document.createElement("img");
      img.src = e.target.result;
      imgEl.appendChild(img);
    };
    reader.readAsDataURL(file);
  } else if (file.type === "application/pdf") {
    imgEl.innerHTML = `<i class="fas fa-file-pdf" style="color:#f97373;font-size:1.5rem;"></i>`;
  } else {
    imgEl.innerHTML = `<i class="fas fa-file-lines" style="color:#facc6b;font-size:1.5rem;"></i>`;
  }

  toast("Media attached for cross‑modal analysis.", "info");
}

function removeFile() {
  currentFile = null;
  document.getElementById("filePreview").classList.add("hidden");
  document.getElementById("uploadArea").style.display = "block";
  document.getElementById("fileInput").value = "";
  toast("Attachment removed.", "info");
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// === ANALYSIS CALL ===
async function analyzeContent() {
  if (!currentFile) {
    toast("Attach at least one media/text file before running analysis.", "error");
    return;
  }

  const text = document.getElementById("textContext").value.trim();
  const lang = document.getElementById("languageSelect").value || "en";

  const btn = document.getElementById("analyzeBtn");
  const main = btn.querySelector(".btn-main");
  const loader = btn.querySelector(".btn-loader");
  btn.disabled = true;
  main.style.display = "none";
  loader.style.display = "inline-flex";

  try {
    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("text_context", text || "No text");
    formData.append("language", lang);

    const res = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error("Backend error " + res.status);
    }
    const data = await res.json();
    analysisResults = data;
    renderAll(data);
    toast("Analysis complete. Evidence chain updated.", "success");
  } catch (err) {
    console.error(err);
    toast("Could not reach analysis server. Check backend.", "error");
  } finally {
    btn.disabled = false;
    main.style.display = "inline-flex";
    loader.style.display = "none";
  }
}

// === RENDERING ===
function renderAll(data) {
  renderOverview(data);
  renderTextual(data.textual || {});
  renderVisual(data.visual || {});
  renderSource(data.source || {});
  renderEvidence(data.evidence_chain || []);
}

function renderOverview(data) {
  const score = data.overall_risk_score || 0;
  const riskEl = document.getElementById("riskScore");
  const gauge = document.getElementById("gaugeProgress");
  const verdictBadge = document.getElementById("verdictBadge");
  const textMini = document.getElementById("textRiskMini");
  const visualMini = document.getElementById("visualRiskMini");
  const sourceMini = document.getElementById("sourceRiskMini");
  const summaryEl = document.getElementById("narrativeSummary");

  animateNumber(riskEl, 0, score, 1800);
  const circ = 251.2;
  const offset = circ - (score / 100) * circ;
  setTimeout(() => {
    gauge.style.transition = "stroke-dashoffset 1.4s ease-out";
    gauge.style.strokeDashoffset = offset;
  }, 80);

  let verdictText = data.verdict || "LOW RISK";
  verdictBadge.textContent = verdictText;
  verdictBadge.className = "badge badge-soft";

  if (score >= 70) {
    verdictBadge.style.borderColor = "rgba(249,115,115,0.9)";
    verdictBadge.style.color = "#fecaca";
  } else if (score >= 40) {
    verdictBadge.style.borderColor = "rgba(250,204,21,0.9)";
    verdictBadge.style.color = "#facc15";
  } else {
    verdictBadge.style.borderColor = "rgba(34,197,94,0.9)";
    verdictBadge.style.color = "#bbf7d0";
  }

  const textRisk = 100 - (data.textual?.credibility_score || 0);
  const visualRisk = data.visual?.deepfake_score || 0;
  const sourceRisk = 100 - (data.source?.trust_score || 0);
  textMini.textContent = isNaN(textRisk) ? "–" : textRisk.toFixed(1) + "%";
  visualMini.textContent = visualRisk.toFixed(1) + "%";
  sourceMini.textContent = isNaN(sourceRisk) ? "–" : sourceRisk.toFixed(1) + "%";

  const narrative =
    score >= 70
      ? "Signals suggest a high‑risk case. Language, visuals, and sourcing should be scrutinized manually before amplification."
      : score >= 40
      ? "Signals are mixed. Some modalities raise concerns, but corroborating evidence exists. Treat with caution."
      : "Most modalities agree with authentic patterns. Low‑risk, though critical claims still warrant human judgment.";
  summaryEl.textContent = narrative;

  const timeline = document.getElementById("timelineSteps");
  timeline.innerHTML = "";
  (data.evidence_chain || []).forEach((ev) => {
    const step = document.createElement("div");
    step.className = "timeline-step";
    const fill = document.createElement("div");
    fill.className = "timeline-step-fill";
    fill.style.background =
      ev.type === "textual"
        ? "linear-gradient(90deg,#facc6b,#f97316)"
        : ev.type === "visual"
        ? "linear-gradient(90deg,#f97373,#f97316)"
        : "linear-gradient(90deg,#22c55e,#16a34a)";
    step.appendChild(fill);
    timeline.appendChild(step);
    requestAnimationFrame(() => {
      fill.style.transform = "scaleX(" + Math.min(ev.weight * 1.3, 1) + ")";
    });
  });
}

function renderTextual(t) {
  const cred = t.credibility_score || 0;
  const sens = t.sensationalism_index || 0;
  const bias = (t.political_bias || 0) * 100;
  const sent = (t.sentiment_score || 0) * 100;

  updateBar("credibilityBar", "credibilityScore", cred, "%");
  updateBar("sensationalismBar", "sensationalismScore", sens * 10, "/10", sens.toFixed(1));
  updateBar("biasBar", "biasScore", bias, "%");
  updateBar("sentimentBar", "sentimentScore", sent, "%");

  const snippetEl = document.getElementById("highlightedText");
  const ctx = document.getElementById("textContext").value || "";
  if (!ctx) {
    snippetEl.textContent =
      "Once you provide text, the model will highlight emotionally loaded segments and all‑or‑nothing claims.";
    return;
  }
  const keywords = ["guaranteed", "never", "always", "shocking", "breaking", "exposed"];
  let snippet = ctx.slice(0, 260);
  keywords.forEach((k) => {
    const reg = new RegExp("\\b" + k + "\\b", "gi");
    snippet = snippet.replace(reg, (m) => `<mark>${m}</mark>`);
  });
  snippetEl.innerHTML = snippet + (ctx.length > 260 ? "…" : "");
}

function renderVisual(v) {
  const deep = v.deepfake_score || 0;
  const conf = (v.confidence || 0) * 100;
  const verdict = v.verdict || "N/A";

  const deepEl = document.getElementById("deepfakeScore");
  const confEl = document.getElementById("confidenceScore");
  const verEl = document.getElementById("consistencyVerdict");
  const ring = document.getElementById("deepfakeRing");

  deepEl.textContent = deep.toFixed(1) + "%";
  confEl.textContent = conf.toFixed(0) + "%";
  verEl.textContent = verdict;

  if (deep >= 70) {
    ring.style.boxShadow = "0 0 30px rgba(249,115,115,0.7)";
  } else if (deep >= 40) {
    ring.style.boxShadow = "0 0 22px rgba(250,204,21,0.7)";
  } else {
    ring.style.boxShadow = "0 0 22px rgba(34,197,94,0.55)";
  }

  const artifacts = v.artifacts || [];
  const list = document.getElementById("artifactsList");
  list.innerHTML = "";
  if (!artifacts.length) {
    list.innerHTML = `<span class="artifact-tag">No obvious artifacts surfaced by the model.</span>`;
  } else {
    artifacts.forEach((a) => {
      const span = document.createElement("span");
      span.className = "artifact-tag";
      span.textContent = a;
      list.appendChild(span);
    });
  }
}

function renderSource(s) {
  const trust = s.trust_score || 0;
  updateBar("trustBar", "trustScore", trust, "%");

  const cont = document.getElementById("sourcesContainer");
  cont.innerHTML = "";
  const links = s.links || [];
  if (!links.length) {
    cont.innerHTML =
      '<div class="sources-empty">No corroborating sources found for this snippet.</div>';
    return;
  }
  links.forEach((lnk) => {
    const item = document.createElement("a");
    item.href = lnk.url;
    item.target = "_blank";
    item.rel = "noopener noreferrer";
    item.className = "source-item";

    const icon = document.createElement("div");
    icon.className = "source-icon";
    const i = document.createElement("i");
    i.className = lnk.is_trusted ? "fas fa-check-circle trusted" : "fas fa-circle untrusted";
    icon.appendChild(i);

    const main = document.createElement("div");
    main.className = "source-main";
    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = lnk.title || "Untitled source";
    const url = document.createElement("div");
    url.className = "source-url";
    url.textContent = lnk.url;

    main.appendChild(title);
    main.appendChild(url);

    item.appendChild(icon);
    item.appendChild(main);
    cont.appendChild(item);
  });
}

function renderEvidence(chain) {
  const container = document.getElementById("evidenceChain");
  container.innerHTML = "";
  if (!chain.length) {
    container.innerHTML =
      '<div class="note">Evidence weights will appear here after the first analysis.</div>';
    return;
  }
  chain.forEach((ev) => {
    const card = document.createElement("div");
    card.className = "evidence-card";
    const top = document.createElement("div");
    top.className = "evidence-top";
    const type = document.createElement("div");
    type.className = "evidence-type";
    type.textContent = ev.type.toUpperCase();
    const w = document.createElement("div");
    w.className = "evidence-weight";
    w.textContent = `${(ev.weight * 100).toFixed(0)}% weight`;
    top.appendChild(type);
    top.appendChild(w);

    const score = document.createElement("div");
    score.className = "evidence-score";
    score.style.color = ev.score > 50 ? "#f97316" : "#22c55e";
    score.textContent = ev.score.toFixed(1) + "%";

    const reason = document.createElement("div");
    reason.className = "evidence-reason";
    reason.textContent = ev.reason;

    card.appendChild(top);
    card.appendChild(score);
    card.appendChild(reason);
    container.appendChild(card);
  });
}

// === HELPERS ===
function updateBar(barId, labelId, percent, suffix, forcedLabel) {
  const bar = document.getElementById(barId);
  const label = document.getElementById(labelId);
  if (!bar || !label) return;
  const p = Math.max(0, Math.min(100, percent || 0));
  bar.style.width = "0%";
  setTimeout(() => {
    bar.style.width = p + "%";
  }, 60);
  label.textContent = (forcedLabel || p.toFixed(1)) + suffix;
}

function animateNumber(el, start, end, duration) {
  const range = end - start;
  let current = start;
  const stepTime = 16;
  const steps = duration / stepTime;
  const inc = range / steps;
  const timer = setInterval(() => {
    current += inc;
    if ((inc > 0 && current >= end) || (inc < 0 && current <= end)) {
      current = end;
      clearInterval(timer);
    }
    el.textContent = Math.round(current);
  }, stepTime);
}

let toastTimer;
function toast(msg, type = "info") {
  const t = document.getElementById("toast");
  const msgEl = document.getElementById("toastMessage");
  const icon = t.querySelector("i");
  msgEl.textContent = msg;
  if (type === "success") {
    icon.className = "fas fa-circle-check";
    t.style.borderColor = "rgba(34,197,94,0.85)";
  } else if (type === "error") {
    icon.className = "fas fa-circle-exclamation";
    t.style.borderColor = "rgba(249,115,115,0.9)";
  } else {
    icon.className = "fas fa-circle-info";
    t.style.borderColor = "rgba(250,204,21,0.9)";
  }
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.classList.remove("show");
  }, 2600);
}
