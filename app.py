#!/usr/bin/env python
# ============================================================================
# MediaShield - Multimodal Intelligence Hub Backend
# Features: SerpApi, BERT Fake-News, CLIP Consistency, PDF Export, CORS
# ============================================================================

import os
import uuid
import logging
import re
from datetime import datetime
from io import BytesIO

import torch
from PIL import Image
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from transformers import pipeline, CLIPProcessor, CLIPModel

# Search API
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
except ImportError:
    SimpleDocTemplate = None

# Load .env
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "txt", "pdf"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SERPAPI_KEY = os.getenv("SERPAPI_KEY")

    TRUSTED_DOMAINS = [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "npr.org",
        "pbs.org",
        "wsj.com",
        "nytimes.com",
        "theguardian.com",
        "dw.com",
        "france24.com",
        "bloomberg.com",
        "snopes.com",
        "politifact.com",
        "thehindu.com",
        "indianexpress.com",
        "timesofindia.indiatimes.com",
        "hindustantimes.com",
        "ndtv.com",
        "livemint.com",
        "business-standard.com",
    ]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mediashield")

if not Config.SERPAPI_KEY:
    logger.warning("⚠️  SERPAPI_KEY not found in .env. Source verification will be degraded.")

# Ensure upload folder exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ============================================================================
# 1. TEXT FORENSICS ENGINE (BERT)
# ============================================================================

class RealTextAnalyzer:
    def __init__(self):
        logger.info("Loading Text Forensics Model (BERT-Tiny Fake News)...")
        device = 0 if Config.DEVICE == "cuda" else -1

        # Fake news head
        self.classifier = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-fake-news-detection",
            device=device,
        )  # [web:54]

        # Sentiment head
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )

    def analyze(self, text: str) -> dict:
        try:
            safe_text = text[:512] if text else "No text"
            result = self.classifier(safe_text)[0]
            label = result["label"]
            score = float(result["score"])

            if label.upper() == "FAKE":
                risk_score = score * 100.0
            else:
                risk_score = (1.0 - score) * 100.0

            sent_result = self.sentiment(safe_text)[0]
            sent_score = float(sent_result["score"])
            sent_label = sent_result["label"]

            sensationalism = sent_score * 10.0 if sent_label == "NEGATIVE" else 2.0

            return {
                "credibility_verdict": label,
                "model_confidence": round(score * 100.0, 2),
                "risk_score": round(risk_score, 2),
                "sensationalism_index": round(sensationalism, 1),
            }
        except Exception as e:
            logger.error(f"Text Analysis Failed: {e}")
            return {
                "risk_score": 50.0,
                "sensationalism_index": 5.0,
                "model_confidence": 50.0,
                "credibility_verdict": "ERROR",
                "error": str(e),
            }

# ============================================================================
# 2. MULTIMODAL FORENSICS (CLIP)
# ============================================================================

class RealMultimodalAnalyzer:
    def __init__(self):
        logger.info("Loading Multimodal Model (openai/clip-vit-base-patch32)...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            Config.DEVICE
        )  # [web:59]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def check_consistency(self, image_path: str, text_headline: str) -> dict:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=[text_headline[:77]],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(Config.DEVICE)

            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image.item() / 100.0

            consistency_score = min(max((similarity - 0.15) * 500, 0), 100)
            risk_score = 100.0 - consistency_score

            verdict = "MATCH" if consistency_score > 40.0 else "MISMATCH"
            return {
                "consistency_score": round(consistency_score, 2),
                "risk_score": round(risk_score, 2),
                "verdict": verdict,
            }
        except Exception as e:
            logger.error(f"CLIP Analysis Failed: {e}")
            return {
                "consistency_score": 0.0,
                "risk_score": 0.0,
                "verdict": "N/A",
                "error": str(e),
            }

# ============================================================================
# 3. SOURCE VERIFICATION (SERPAPI)
# ============================================================================

class RealSourceVerifier:
    def verify(self, text_segment: str) -> dict:
        if not text_segment:
            return {
                "risk_score": 60.0,
                "trusted_sources_count": 0,
                "links": [],
                "found_trusted": False,
            }

        clean_text = re.sub(r"--- PAGE \d+ ---", "", text_segment)
        clean_text = re.sub(r"\[.*?\]", "", clean_text)
        clean_text = re.sub(r"[^\w\s]", " ", clean_text)
        clean_text = " ".join(clean_text.split())

        tokens = clean_text.split()
        query_long = " ".join(tokens[:12])
        query_short = " ".join(tokens[:6])

        logger.info(f"[SerpApi] Primary query: '{query_long}'")
        results = self._perform_search(query_long)

        if not results.get("found_trusted"):
            logger.info(f"[SerpApi] Fallback query: '{query_short}'")
            fallback = self._perform_search(query_short)
            # Favor whichever has more trusted hits
            if fallback.get("trusted_sources_count", 0) > results.get(
                "trusted_sources_count", 0
            ):
                results = fallback

        return results

    def _perform_search(self, query: str) -> dict:
        if not Config.SERPAPI_KEY or GoogleSearch is None:
            return {
                "risk_score": 50.0,
                "error": "SerpAPI not configured",
                "found_trusted": False,
                "trusted_sources_count": 0,
                "links": [],
            }

        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": Config.SERPAPI_KEY,
                "num": 5,
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            found_sources = []
            trusted_count = 0

            for result in organic_results:
                link = result.get("link", "")
                title = result.get("title", "")
                domain_trust = any(
                    trusted in link.lower() for trusted in Config.TRUSTED_DOMAINS
                )
                if domain_trust:
                    trusted_count += 1

                found_sources.append(
                    {"title": title, "url": link, "is_trusted": domain_trust}
                )

            if trusted_count >= 1:
                risk_score = 0.0
            elif len(found_sources) > 0:
                risk_score = 40.0
            else:
                risk_score = 90.0

            return {
                "risk_score": risk_score,
                "trusted_sources_count": trusted_count,
                "links": found_sources,
                "found_trusted": trusted_count > 0,
            }
        except Exception as e:
            logger.error(f"SerpApi Error: {e}")
            return {
                "risk_score": 50.0,
                "error": str(e),
                "found_trusted": False,
                "trusted_sources_count": 0,
                "links": [],
            }

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# CORS for /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})  # [web:56][web:60]

logger.info("Initializing AI Models...")
text_engine = RealTextAnalyzer()
multimodal_engine = RealMultimodalAnalyzer()
source_engine = RealSourceVerifier()
logger.info("Models ready.")

# --------------------------------------------------------------------------
# Helper: extension check
# --------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
  return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# Health check for frontend
# --------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """Simple health endpoint so frontend can check status."""
    return jsonify(
        {
            "status": "ok",
            "message": "MediaShield backend online",
            "device": Config.DEVICE,
        }
    ), 200

# --------------------------------------------------------------------------
# MAIN ANALYSIS ENDPOINT (used by script.js)
# --------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    text_content = request.form.get("text_context", "").strip()
    language = request.form.get("language", "en")

    if not text_content and filename.lower().endswith(".pdf"):
        text_content = "Extracted text placeholder for PDF analysis."

    # 1) TEXT ANALYSIS (BERT)
    text_results = text_engine.analyze(text_content if text_content else "No text")

    # 2) VISUAL CONSISTENCY (CLIP) if image + text
    if filename.lower().endswith((".png", ".jpg", ".jpeg")) and text_content:
        visual_results = multimodal_engine.check_consistency(save_path, text_content)
    else:
        visual_results = {
            "consistency_score": 0.0,
            "risk_score": 0.0,
            "verdict": "N/A",
        }

    # 3) SOURCE VERIFICATION (SerpAPI)
    search_input = text_content[:200]
    if filename.lower().endswith(".pdf"):
        clean_fname = filename.replace("-", " ").replace(".pdf", "")
        search_input = f"{clean_fname} {text_content[:100]}"
    source_results = source_engine.verify(search_input)

    # 4) WEIGHTED OVERALL SCORE (matches frontend expectations)
    final_score = (
        text_results.get("risk_score", 50.0) * 0.4
        + visual_results.get("risk_score", 0.0) * 0.3
        + source_results.get("risk_score", 50.0) * 0.3
    )

    overall_risk = round(final_score, 1)
    verdict = "HIGH RISK" if overall_risk > 65 else "LOW RISK"

    response = {
        "overall_risk_score": overall_risk,
        "deepfake_probability": round(visual_results.get("risk_score", 0.0), 1),
        "detected_language": language,
        "timestamp": datetime.utcnow().isoformat(),
        "analysis_id": str(uuid.uuid4()),
        "status": "success",
        "verdict": verdict,
        # visual block used by frontend
        "visual": {
            "deepfake_score": visual_results.get("risk_score", 0.0),
            "confidence": 0.94,  # static for now
            "artifacts": [],     # can be filled by later models
            "verdict": visual_results.get("verdict", "N/A"),
        },
        # textual block used by frontend
        "textual": {
            "sensationalism_index": text_results.get("sensationalism_index", 5.0),
            "credibility_score": 100.0 - text_results.get("risk_score", 50.0),
            "political_bias": 0.5,   # placeholder, UI expects float 0–1
            "sentiment_score": 0.5,  # placeholder, UI expects float 0–1
        },
        # source block used by frontend
        "source": {
            "trust_score": 100.0 - source_results.get("risk_score", 50.0),
            "links": source_results.get("links", []),
        },
        # evidence chain for bottom cards & timeline
        "evidence_chain": [
            {
                "type": "textual",
                "score": text_results.get("risk_score", 0.0),
                "weight": 0.40,
                "reason": f"BERT verdict: {text_results.get('credibility_verdict', 'N/A')}",
            },
            {
                "type": "visual",
                "score": visual_results.get("risk_score", 0.0),
                "weight": 0.30,
                "reason": f"CLIP verdict: {visual_results.get('verdict', 'N/A')}",
            },
            {
                "type": "source",
                "score": source_results.get("risk_score", 50.0),
                "weight": 0.30,
                "reason": f"Trusted sources: {source_results.get('trusted_sources_count', 0)}",
            },
        ],
        "recommendation": (
            "HIGH RISK: Flag for manual review"
            if overall_risk > 65
            else "LOW RISK: Likely authentic content"
        ),
    }

    return jsonify(response), 200

# --------------------------------------------------------------------------
# PDF EXPORT (used by export button in older frontend; you can still keep it)
# --------------------------------------------------------------------------
@app.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    if SimpleDocTemplate is None:
        return jsonify({"error": "PDF generation unavailable (reportlab not installed)"}), 500

    try:
        payload = request.get_json() or {}
        pdf_buffer = BytesIO()
        styles = getSampleStyleSheet()

        COLOR_PRIMARY = colors.HexColor("#0f172a")
        COLOR_ACCENT = colors.HexColor("#facc6b")
        COLOR_BG_GRAY = colors.HexColor("#f9fafb")

        style_normal = ParagraphStyle(
            "CustomNormal",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#374151"),
            leading=14,
        )
        style_meta = ParagraphStyle(
            "MetaData",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.gray,
            alignment=2,
        )
        style_h2 = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=COLOR_PRIMARY,
            spaceBefore=20,
            spaceAfter=10,
            borderPadding=5,
            borderColor=COLOR_ACCENT,
            borderWidth=0,
            borderBottomWidth=1,
        )

        def header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(COLOR_PRIMARY)
            canvas.rect(0, 10.5 * inch, 8.5 * inch, 0.5 * inch, fill=1, stroke=0)

            canvas.setFillColor(colors.white)
            canvas.setFont("Helvetica-Bold", 14)
            canvas.drawString(0.5 * inch, 10.65 * inch, "MediaShield | Forensic Analysis")

            canvas.setFont("Helvetica", 9)
            canvas.drawRightString(8 * inch, 10.65 * inch, "CONFIDENTIAL REPORT")

            canvas.setStrokeColor(colors.lightgrey)
            canvas.line(0.5 * inch, 0.75 * inch, 8 * inch, 0.75 * inch)
            canvas.setFillColor(colors.gray)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(
                0.5 * inch,
                0.5 * inch,
                f"Generated via MediaShield AI • Page {doc.page}",
            )
            canvas.restoreState()

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=70,
            bottomMargin=70,
        )

        elements = []

        analysis_id = payload.get("analysis_id", "N/A")
        timestamp = payload.get(
            "timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        )

        meta_data = [
            [
                Paragraph(f"<b>Analysis ID:</b> {analysis_id}", style_normal),
                Paragraph(f"<b>Date:</b> {timestamp}", style_meta),
            ]
        ]
        meta_table = Table(meta_data, colWidths=[4 * inch, 2.5 * inch])
        meta_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 20))

        risk_score = float(payload.get("overall_risk_score", 0.0))
        if risk_score >= 70:
            risk_bg = colors.HexColor("#ef4444")
            risk_text = "HIGH RISK"
        elif risk_score >= 40:
            risk_bg = colors.HexColor("#eab308")
            risk_text = "MEDIUM RISK"
        else:
            risk_bg = colors.HexColor("#22c55e")
            risk_text = "LOW RISK"

        risk_data = [
            ["OVERALL RISK SCORE", f"{risk_score:.1f}%"],
            [risk_text, "Probability of Misinformation"],
        ]
        t_risk = Table(risk_data, colWidths=[3.25 * inch, 3.25 * inch])
        t_risk.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 1), COLOR_PRIMARY),
                    ("BACKGROUND", (1, 0), (1, 1), risk_bg),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 10),
                    ("FONTSIZE", (1, 0), (1, 0), 24),
                    ("FONTSIZE", (0, 1), (0, 1), 16),
                    ("FONTSIZE", (1, 1), (1, 1), 10),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                ]
            )
        )
        elements.append(t_risk)
        elements.append(Spacer(1, 30))

        if "evidence_chain" in payload:
            elements.append(Paragraph("Evidence Chain Analysis", style_h2))
            e_data = [["Evidence Type", "Risk Contribution", "Weight", "Observation"]]
            for ev in payload["evidence_chain"]:
                score_val = float(ev.get("score", 0.0))
                score_color = colors.red if score_val > 50 else colors.black
                row = [
                    Paragraph(f"<b>{ev.get('type', '').upper()}</b>", style_normal),
                    Paragraph(
                        f"<font color={score_color}>{score_val:.1f}%</font>",
                        style_normal,
                    ),
                    f"{ev.get('weight', 0) * 100:.0f}%",
                    Paragraph(ev.get("reason", ""), style_normal),
                ]
                e_data.append(row)
            t_evidence = Table(e_data, colWidths=[1.2 * inch, 1.2 * inch, 0.8 * inch, 3.3 * inch])
            t_evidence.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_GRAY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PRIMARY),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("TOPPADDING", (0, 0), (-1, 0), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            elements.append(t_evidence)
            elements.append(Spacer(1, 20))

        elements.append(Paragraph("Detailed Forensic Breakdown", style_h2))

        txt_stats = ""
        t_block = payload.get("textual", {})
        if t_block:
            txt_stats = f"""
            <b>Sensationalism Index:</b> {t_block.get('sensationalism_index', 0):.1f}/10<br/>
            <b>Credibility Score:</b> {t_block.get('credibility_score', 0):.1f}%<br/>
            <b>Political Bias:</b> {t_block.get('political_bias', 0)*100:.0f}%<br/>
            """

        vis_stats = ""
        v_block = payload.get("visual", {})
        if v_block:
            vis_stats += f"<b>Deepfake Score:</b> {v_block.get('deepfake_score', 0):.1f}%<br/>"
            vis_stats += f"<b>Confidence:</b> {v_block.get('confidence', 0)*100:.0f}%<br/>"
        s_block = payload.get("source", {})
        if s_block:
            vis_stats += f"<b>Source Trust:</b> {s_block.get('trust_score', 0):.1f}/100<br/>"

        details_table_data = [
            [
                Paragraph("<b>Textual Forensics</b>", style_normal),
                Paragraph("<b>Visual & Source Forensics</b>", style_normal),
            ],
            [Paragraph(txt_stats, style_normal), Paragraph(vis_stats, style_normal)],
        ]
        t_details = Table(details_table_data, colWidths=[3.25 * inch, 3.25 * inch])
        t_details.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_GRAY),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        elements.append(t_details)
        elements.append(Spacer(1, 20))

        if "recommendation" in payload:
            rec_text = payload["recommendation"]
            rec_color = (
                colors.HexColor("#dcfce7")
                if "LOW" in rec_text
                else colors.HexColor("#fee2e2")
            )
            rec_border = (
                colors.HexColor("#22c55e")
                if "LOW" in rec_text
                else colors.HexColor("#ef4444")
            )
            rec_para = Paragraph(
                f"<b>FINAL RECOMMENDATION:</b><br/>{rec_text}", style_normal
            )
            t_rec = Table([[rec_para]], colWidths=[6.5 * inch])
            t_rec.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), rec_color),
                        ("BOX", (0, 0), (-1, -1), 1, rec_border),
                        ("TOPPADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ]
                )
            )
            elements.append(t_rec)

        doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

        pdf_buffer.seek(0)
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"MediaShield_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("MediaShield Server Running on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
