# Multimodal Deepfake Forensic Analyzer ✅

A compact, browser-based frontend and Python backend for forensic analysis of images, video, and text to detect signs of manipulation, provenance issues, and credibility concerns.

---

## 🚀 Overview

**Multimodal Deepfake Forensic Analyzer** is a lightweight demo that combines visual, textual, and provenance analysis to estimate the risk that a piece of media is manipulated or being used out-of-context. It provides a clear verdict, an overall risk score, chain-of-evidence, and an explainable reasoning component.

Key features:

- Visual deepfake detection and artifact listing
- Textual credibility & sensationalism scoring
- Source corroboration and trust scoring
- Provenance timeline and context-hijacking alerts
- Human-friendly UI with attention highlights and chain-of-thought reasoning

---

## 🔧 Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies

---

## Get Started (Quick)

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend API (default port: 5001):

```bash
python app_enhanced_PATCHED.py
```

4. Open `index.html` in your browser or serve the folder (e.g. `python -m http.server 8000`) and visit `http://localhost:8000`.

> The front-end JavaScript expects the backend at `http://localhost:5001` (see `script.js` → `API_BASE_URL`).

---

## 📡 API (examples)

- Health check:

```bash
curl http://localhost:5001/health
```

- Analyze (multipart/form-data):

```bash
curl -X POST http://localhost:5001/analyze \
  -F "file=@path/to/media.jpg" \
  -F "text=Optional surrounding text" \
  -F "language=en"
```

Example response (truncated):

```json
{
  "overall_risk_score": 62.1,
  "verdict": "MEDIUM RISK",
  "textual": {"credibility_score": 58.3},
  "visual": {"deepfake_score": 72.4, "artifacts": ["face-blend", "temporal-inconsistency"]},
  "provenance": {"timeline": [...], "context_hijacking_detected": true},
  "reasoning": {"explanation": "...", "method": "llm-cot"},
  "evidence_chain": [ ... ]
}
```

---

## 🧩 Frontend notes

- Main UI is `index.html` and `script.js`.
- To change backend address, edit `API_BASE_URL` at top of `script.js`.
- The UI supports: image, PDF, MP4, and plain text uploads (max 50MB by default).

---

## 🧪 Tests & Debug

- Quick environment check: `python test_env.py`
- If the front-end shows "⚠ Backend not responding", ensure the backend is running and reachable at the configured `API_BASE_URL` and port.

---

## ⚠️ Security & Responsible Use

This repository is intended for research, education, and defensive use. Deepfake detection is an evolving field — results are indicative, not definitive. Do not rely solely on this tool for high-stakes decisions. Respect privacy and applicable laws when analyzing third-party media.

---

## Contributing

Contributions are welcome. Open issues for bugs or feature requests, and submit pull requests with tests and documentation updates.

---

## License

See `LICENSE` in the repository.

---

## Acknowledgements

Built as a multimodal forensic demo combining visual, textual, provenance, and explainability ideas. Thank you to contributors and open-source projects that make this possible.

---

If you want, I can also add usage examples to `index.html` or create a minimal Dockerfile for easy deployment. ✨
