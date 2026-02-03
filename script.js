// CONFIG
const API_BASE_URL = "http://localhost:5001";

let currentFile = null;
let analysisResults = null;

document.addEventListener("DOMContentLoaded", () => {
  drawLines();
  initUpload();
  initEventStream();
  initNotes();
  initThemeToggle();
  checkBackendHealth();
});

// === CHECK BACKEND HEALTH ===
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      updateEventStream("✓ Backend online - Enhanced features loaded");
      updateEventStream(`✓ Device: ${data.device}`);
      if (data.features) {
        updateEventStream(
          `✓ Provenance: ${data.features.provenance ? "ON" : "OFF"}`
        );
        updateEventStream(
          `✓ Deepfake AI: ${data.features.deepfake ? "ON" : "OFF"}`
        );
        updateEventStream(
          `✓ CoT Reasoning: ${data.features.cot ? "ON" : "OFF"}`
        );
        updateEventStream(
          `✓ Multilingual: ${data.features.multilingual ? "ON" : "OFF"}`
        );
      }
    }
  } catch (e) {
    updateEventStream("⚠ Backend not responding");
  }
}

function initEventStream() {
  const stream = document.getElementById("eventStream");
  if (stream) {
    stream.innerHTML = "";
  }
}

function updateEventStream(message) {
  const stream = document.getElementById("eventStream");
  if (!stream) return;
  const div = document.createElement("div");
  div.className = "mini-log-item";
  div.textContent = message;
  stream.appendChild(div);
  // Keep only last 6 messages
  while (stream.children.length > 6) {
    stream.removeChild(stream.firstChild);
  }
}

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

// === NOTES ===
function initNotes() {
  const list = document.getElementById("noteList");
  if (!list) return;
  const notes = [
    "Provenance tracking detects when images are reused in different contexts (context hijacking).",
    "Attention visualization shows which words the AI model focused on during analysis.",
    "Chain-of-Thought reasoning provides step-by-step explanations for AI verdicts.",
    "Deepfake detection uses neural networks trained on manipulated media datasets.",
    "Multilingual support enables analysis of content in 100+ languages.",
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
  uploadArea.addEventListener("dragover", (e) => e.preventDefault());
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
    "video/mp4",
  ];
  if (file.type && !allowed.includes(file.type)) {
    toast("Unsupported file type. Use JPG, PNG, TXT, PDF, or MP4.", "error");
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    toast("File larger than 50MB not allowed.", "error");
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

  upload.style.display = "none";
  preview.classList.remove("hidden");

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
  } else if (file.type === "video/mp4") {
    imgEl.innerHTML = `<i class="fas fa-film" style="color:#f97373;font-size:1.5rem;"></i>`;
  } else if (file.type === "application/pdf") {
    imgEl.innerHTML = `<i class="fas fa-file-pdf" style="color:#f97373;font-size:1.5rem;"></i>`;
  } else {
    imgEl.innerHTML = `<i class="fas fa-file-lines" style="color:#facc6b;font-size:1.5rem;"></i>`;
  }

  toast("Media ready for forensic analysis.", "info");
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
    toast("Attach media before running analysis.", "error");
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

  updateEventStream("🔬 Starting enhanced analysis...");

  try {
    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("text", text || "No text");
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
    toast("Analysis complete with AI reasoning!", "success");
    updateEventStream("✓ Analysis complete");
  } catch (err) {
    console.error(err);
    toast("Analysis failed. Check backend connection.", "error");
    updateEventStream("✗ Analysis failed");
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
  renderProvenance(data.provenance || {});
  renderReasoning(data.reasoning || {});
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
  const provenanceMini = document.getElementById("provenanceRiskMini");
  const summaryEl = document.getElementById("narrativeSummary");

  animateNumber(riskEl, 0, score, 1800);
  const circ = 251.2;
  const offset = circ - (score / 100) * circ;
  setTimeout(() => {
    gauge.style.transition = "stroke-dashoffset 1.4s ease-out";
    gauge.style.strokeDashoffset = offset;
  }, 80);

  verdictBadge.textContent = data.verdict || "LOW RISK";
  verdictBadge.className = "badge badge-soft";

  if (score >= 65) {
    verdictBadge.style.borderColor = "rgba(249,115,115,0.9)";
    verdictBadge.style.color = "#fecaca";
  } else if (score >= 35) {
    verdictBadge.style.borderColor = "rgba(250,204,21,0.9)";
    verdictBadge.style.color = "#facc15";
  } else {
    verdictBadge.style.borderColor = "rgba(34,197,94,0.9)";
    verdictBadge.style.color = "#bbf7d0";
  }

  const textRisk = 100 - (data.textual?.credibility_score || 0);
  const visualRisk = data.visual?.deepfake_score || 0;
  const sourceRisk = 100 - (data.source?.trust_score || 0);
  const provenanceRisk = data.provenance?.risk_score || 0;

  textMini.textContent = textRisk.toFixed(1) + "%";
  visualMini.textContent = visualRisk.toFixed(1) + "%";
  sourceMini.textContent = sourceRisk.toFixed(1) + "%";
  provenanceMini.textContent = provenanceRisk.toFixed(1) + "%";

  summaryEl.textContent = data.recommendation || "Analysis in progress...";

  // Timeline
  const timeline = document.getElementById("timelineSteps");
  timeline.innerHTML = "";
  (data.evidence_chain || []).forEach((ev) => {
    const step = document.createElement("div");
    step.className = "timeline-step";
    const fill = document.createElement("div");
    fill.className = "timeline-step-fill";

    const colors = {
      textual: "linear-gradient(90deg,#facc6b,#f97316)",
      visual: "linear-gradient(90deg,#f97373,#ef4444)",
      source: "linear-gradient(90deg,#22c55e,#16a34a)",
      provenance: "linear-gradient(90deg,#60a5fa,#3b82f6)",
    };
    fill.style.background = colors[ev.type] || colors.textual;

    step.appendChild(fill);
    timeline.appendChild(step);
    requestAnimationFrame(() => {
      fill.style.transform = "scaleX(" + Math.min(ev.weight * 1.5, 1) + ")";
    });
  });
}

function renderTextual(t) {
    const cred = t.credibility_score || 0;
    const sens = t.sensationalism_index || 0;

    updateBar("credibilityBar", "credibilityScore", cred, "%");
    updateBar(
        "sensationalismBar",
        "sensationalismScore",
        sens * 10,
        "/10",
        sens.toFixed(1)
    );

    // Language badge
    const langBadge = document.getElementById("languageBadge");
    const lang = t.language_detected || "en";
    langBadge.textContent = `${getLangName(lang)} (${lang})`;

    // Attention highlights - FIXED
    const attentionWords = document.getElementById("attentionWords");
    attentionWords.innerHTML = "";
    const highlights = t.attention_highlights || [];
    
    if (highlights.length > 0) {
        highlights.slice(0, 8).forEach((item) => {
            // Handle both object {word, score} and array [word, score] formats
            const word = item.word || item[0] || "";
            const score = item.score || item[1] || 0;
            
            if (word) {
                const span = document.createElement("span");
                span.className = "attention-word";
                span.textContent = `${word} (${(score * 100).toFixed(0)}%)`;
                span.style.opacity = 0.4 + score * 0.6;
                attentionWords.appendChild(span);
            }
        });
    }

    // Highlighted phrases in text
    const snippetEl = document.getElementById("highlightedText");
    const ctx = document.getElementById("textContext").value || "";
    const keywords = t.highlighted_phrases || ["breaking", "shocking", "urgent"];
    
    if (ctx) {
        let snippet = ctx.slice(0, 260);
        keywords.forEach((k) => {
            const reg = new RegExp("\\b" + k + "\\b", "gi");
            snippet = snippet.replace(reg, (m) => `<mark>${m}</mark>`);
        });
        snippetEl.innerHTML = snippet + (ctx.length > 260 ? "…" : "");
    }
}


function renderVisual(v) {
  // Handle both old and new data structures
  const deep = v.deepfake_score || v.deepfake?.probability || 0;
  const conf = v.confidence || (v.deepfake?.confidence || 0);
  const verdict = v.verdict || v.deepfake?.verdict || "N/A";
  const artifacts = v.artifacts || v.deepfake?.artifacts || [];

  const deepEl = document.getElementById("deepfakeScore");
  const confEl = document.getElementById("confidenceScore");
  const verEl = document.getElementById("consistencyVerdict");
  const ring = document.getElementById("deepfakeRing");

  if (deepEl) deepEl.textContent = deep.toFixed(1) + "%";
  if (confEl) confEl.textContent = (conf * 100).toFixed(0) + "%";
  if (verEl) verEl.textContent = verdict;

  if (ring) {
    if (deep >= 70) {
      ring.style.boxShadow = "0 0 30px rgba(249,115,115,0.7)";
    } else if (deep >= 40) {
      ring.style.boxShadow = "0 0 22px rgba(250,204,21,0.7)";
    } else {
      ring.style.boxShadow = "0 0 22px rgba(34,197,94,0.55)";
    }
  }

  const list = document.getElementById("artifactsList");
  if (list) {
    list.innerHTML = "";
    
    if (!artifacts || artifacts.length === 0) {
      list.innerHTML = `<span class="artifact-tag">No manipulation artifacts detected.</span>`;
    } else {
      artifacts.forEach((a) => {
        const span = document.createElement("span");
        span.className = "artifact-tag";
        span.textContent = a;
        list.appendChild(span);
      });
    }
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
      '<div class="sources-empty">No corroborating sources found.</div>';
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
    i.className = lnk.is_trusted
      ? "fas fa-check-circle trusted"
      : "fas fa-circle untrusted";
    icon.appendChild(i);

    const main = document.createElement("div");
    main.className = "source-main";
    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = lnk.title || "Untitled";
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

function renderProvenance(p) {
  const container = document.getElementById("provenanceTimeline");
  container.innerHTML = "";

  const timeline = p.timeline || [];

  if (!timeline.length) {
    container.innerHTML =
      '<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem;">No provenance data available for this media.</div>';
    return;
  }

  // Context hijacking alert
  if (p.context_hijacking_detected) {
    const alert = document.createElement("div");
    alert.className = "context-hijack-alert";
    alert.innerHTML = `
      <strong>⚠ Context Hijacking Detected!</strong><br/>
      This image has appeared in ${
        p.context_changes?.length || 0
      } different contexts across the web.
    `;
    container.appendChild(alert);
  }

  // Timeline items
  timeline.slice(0, 5).forEach((item) => {
    const div = document.createElement("div");
    div.className = "timeline-item";
    div.innerHTML = `
      <div class="timeline-date">${item.date || "Unknown"}</div>
      <div class="timeline-content">
        <div class="timeline-title">${item.context || "No context"}</div>
        <div class="timeline-source">
          ${item.source || "Unknown source"} 
          ${
            item.url
              ? `• <a href="${item.url}" target="_blank" style="color: var(--primary);">View</a>`
              : ""
          }
        </div>
      </div>
    `;
    container.appendChild(div);
  });

  if (timeline.length > 5) {
    const more = document.createElement("div");
    more.style.cssText =
      "text-align: center; color: var(--text-muted); margin-top: 0.5rem; font-size: 0.8rem;";
    more.textContent = `+ ${timeline.length - 5} more appearances`;
    container.appendChild(more);
  }
}

function renderReasoning(r) {
  const box = document.getElementById("reasoningBox");
  const text = document.getElementById("reasoningText");

  const reasoning = r.explanation || "No reasoning available.";
  const confidence = (r.confidence || 0.75) * 100;
  const method =
    r.method === "llm-cot" ? "LLM Chain-of-Thought" : "Rule-based Logic";

  text.innerHTML = `
    <div style="margin-bottom: 0.5rem; font-size: 0.75rem; color: var(--text-muted);">
      Method: ${method} • Confidence: ${confidence.toFixed(0)}%
    </div>
    ${reasoning}
  `;
}

function renderEvidence(chain) {
  const container = document.getElementById("evidenceChain");
  container.innerHTML = "";

  if (!chain.length) {
    container.innerHTML =
      '<div class="note">Evidence weights will appear after analysis.</div>';
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

function getLangName(code) {
  const names = {
    en: "English",
    hi: "Hindi",
    es: "Spanish",
    fr: "French",
    ar: "Arabic",
    zh: "Chinese",
    de: "German",
    pt: "Portuguese",
  };
  return names[code] || "Unknown";
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
