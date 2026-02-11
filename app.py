"""
Flask API for Zero-Shot Deepfake & Misinformation Detection
=========================================================

Production-ready REST API with:
- Image detection → EnhancedZeroShotVisualDetectorV2
- Text detection  → EnhancedTextDetectorV2
- Multimodal fusion → ZeroShotDetectionSystem
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import traceback
import requests
from PIL import Image

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

# FIXED IMPORTS - Use the enhanced detectors
from Phase_2.zero_shot_visual_detector import EnhancedZeroShotVisualDetectorV2
from Phase_2.text_detector import EnhancedTextDetectorV2
from Phase_2.zero_shot_detection_system import ZeroShotDetectionSystem

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

app.config["ALLOWED_EXTENSIONS"] = {
    "image": {"png", "jpg", "jpeg", "webp", "bmp"},
    "audio": {"mp3", "wav", "ogg", "flac", "m4a"},
    "video": {"mp4", "avi", "mov", "mkv", "webm"}
}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------------------------------------------------------
# Load detectors ONCE
# ------------------------------------------------------------------
print("\n🚀 Loading ENHANCED detection engines...")

try:
    visual_detector = EnhancedZeroShotVisualDetectorV2()
    print("✅ Enhanced visual detector loaded")
except Exception as e:
    print(f"❌ Visual detector failed: {e}")
    visual_detector = None

try:
    text_detector = EnhancedTextDetectorV2()
    print("✅ Enhanced text detector loaded")
except Exception as e:
    print(f"❌ Text detector failed: {e}")
    text_detector = None

try:
    multimodal_detector = ZeroShotDetectionSystem()
    print("✅ Multimodal detector loaded")
except Exception as e:
    print(f"❌ Multimodal detector failed: {e}")
    multimodal_detector = None

def to_native(value):
    """Convert numpy / torch scalars to native Python types"""
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return float(value) if isinstance(value, (int, float)) else value

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def allowed_file(filename, modality):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"][modality]

def authenticity_to_misinformation(score: float) -> float:
    """
    Internal: 0 = fake, 1 = real
    Frontend: 100 = fake, 0 = real
    """
    score = max(0.0, min(1.0, score))
    return round((1.0 - score) * 100.0, 1)

# ------------------------------------------------------------------
# Basic routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if (visual_detector or text_detector or multimodal_detector) else "unhealthy",
        "visual_loaded": visual_detector is not None,
        "text_loaded": text_detector is not None,
        "multimodal_loaded": multimodal_detector is not None
    }), 200

# ------------------------------------------------------------------
# IMAGE ONLY (VISUAL DETECTOR)
# ------------------------------------------------------------------
@app.route("/api/detect-image", methods=["POST"])
def detect_image_only():
    if visual_detector is None:
        return jsonify({"error": "Visual detector unavailable"}), 503

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        if not allowed_file(image_file.filename, "image"):
            return jsonify({"error": "Invalid image type"}), 400

        filename = secure_filename(image_file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(path)

        image = Image.open(path).convert("RGB")
        result = visual_detector.analyze_from_image(image)
        os.remove(path)

        return jsonify({
            "verdict": result.verdict,
            "authenticity_score": round(result.score, 3),
            "misinformation_score": authenticity_to_misinformation(result.score),
            "confidence": round(result.confidence, 2),
            "reasoning": result.reasoning,
            "evidence": result.evidence
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# TEXT ONLY (ENHANCED TEXT DETECTOR)
# ------------------------------------------------------------------
@app.route("/api/detect-text", methods=["POST"])
def detect_text_only():
    if text_detector is None:
        return jsonify({"error": "Text detector unavailable"}), 503

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Use enhanced text detector
        result = text_detector.analyze(text)  # Returns TextAnalysisResult

        # Convert score to fake probability
        fake_prob = 1.0 - result.score

        return jsonify({
            "verdict": result.verdict.upper(),
            "fake_probability": round(to_native(fake_prob), 3),
            "misinformation_score": round(to_native(fake_prob) * 100.0, 1),
            "confidence": round(to_native(result.confidence), 2),
            "reasoning": result.reasoning,
            "evidence": result.evidence
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# FULL MULTIMODAL
# ------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def detect():
    if multimodal_detector is None:
        return jsonify({"error": "Detection system unavailable"}), 503

    try:
        text = request.form.get("text")
        image_path = None
        uploaded = []

        if "image" in request.files:
            img = request.files["image"]
            if allowed_file(img.filename, "image"):
                filename = secure_filename(img.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                img.save(image_path)
                uploaded.append(image_path)

        result = multimodal_detector.detect(
            text=text,
            image_path=image_path
        )

        for f in uploaded:
            try:
                os.remove(f)
            except:
                pass

        fake_prob = float(to_native(result.fake_probability))

        return jsonify({
            "verdict": result.verdict,
            "fake_probability": round(fake_prob, 3),
            "misinformation_score": round(fake_prob * 100.0, 1),
            "confidence": float(to_native(result.confidence)),
            "risk_level": result.risk_level,
            "explanation": result.explanation,
            "evidence": result.evidence
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# FRONTEND DASHBOARD ENDPOINT (FIXED + FIELD MAPPING)
# ------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_frontend():
    """
    Frontend dashboard endpoint with ENHANCED detectors
    MAPS enhanced detector output to frontend's expected field names
    """
    try:
        text = request.form.get("text", "").strip()
        file = request.files.get("file")

        visual_result = None
        text_result = None
        uploaded_files = []

        # ---------------- IMAGE ----------------
        if file and allowed_file(file.filename, "image"):
            if visual_detector is None:
                return jsonify({
                    "error": "Visual detector not available",
                    "status": "error"
                }), 503

            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            uploaded_files.append(path)

            image = Image.open(path).convert("RGB")
            visual_result = visual_detector.analyze_from_image(image)

        # ---------------- TEXT (ENHANCED) ----------------
        if text:
            if text_detector is None:
                return jsonify({
                    "error": "Text detector not available",
                    "status": "error"
                }), 503

            text_result = text_detector.analyze(text)  # TextAnalysisResult

        # Cleanup
        for p in uploaded_files:
            try:
                os.remove(p)
            except:
                pass

        if not visual_result and not text_result:
            return jsonify({
                "error": "No valid input provided",
                "status": "error"
            }), 400

        # ---------------- SCORE AGGREGATION ----------------
        scores = []

        if visual_result:
            visual_fake_prob = 1.0 - visual_result.score
            scores.append(visual_fake_prob)

        if text_result:
            text_fake_prob = 1.0 - text_result.score
            scores.append(text_fake_prob)

        overall_risk = round(
            to_native(sum(scores) / len(scores)) * 100.0, 1
        )
        
        # Risk level
        if overall_risk >= 65:
            risk_level = "HIGH"
        elif overall_risk >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ---------------- EVIDENCE CHAIN ----------------
        evidence_chain = []

        if visual_result:
            visual_risk = to_native(1.0 - visual_result.score) * 100.0
            evidence_chain.append({
                "type": "visual",
                "score": round(visual_risk, 1),
                "weight": 0.6,
                "reason": f"Visual detector verdict: {visual_result.verdict}"
            })

        if text_result:
            text_risk = to_native(1.0 - text_result.score) * 100.0
            evidence_chain.append({
                "type": "textual",
                "score": round(text_risk, 1),
                "weight": 0.4,
                "reason": f"Text analysis verdict: {text_result.verdict.upper()}"
            })

        # ---------------- RESPONSE WITH FIELD MAPPING ----------------
        response = {
            "status": "success",
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "evidence_chain": evidence_chain,
            
            # Visual results
            "visual": {
                "misinformation_score": round(
                    to_native(1.0 - visual_result.score) * 100.0, 1
                ),
                "confidence": round(to_native(visual_result.confidence), 2),
                "verdict": visual_result.verdict,
                "evidence": [
                    str(to_native(e)) if not isinstance(e, (str, list, dict)) else e 
                    for e in visual_result.evidence
                ]
            } if visual_result else None,

            # Text results (MAPPED TO FRONTEND FIELDS)
            "textual": {
                # Core scores
                "misinformation_score": round(to_native(1.0 - text_result.score) * 100.0, 1),
                "confidence": round(to_native(text_result.confidence), 2),
                "verdict": text_result.verdict.upper(),
                
                # MAPPED FIELDS for frontend compatibility
                "credibility_score": round(to_native(text_result.score) * 100.0, 1),  # Inverse of misinformation
                "sensationalism_index": round((1.0 - to_native(text_result.score)) * 10.0, 1),  # 0-10 scale
                
                # Language detection (default to English)
                "language_detected": "en",
                
                # Extract attention highlights from evidence
                "attention_highlights": _extract_attention_highlights(text_result.evidence, text),
                
                # Extract highlighted phrases
                "highlighted_phrases": _extract_highlighted_phrases(text_result.evidence),
                
                # Reasoning and evidence
                "reasoning": text_result.reasoning,
                "evidence": [
                    str(to_native(e)) if not isinstance(e, (str, list, dict)) else e 
                    for e in text_result.evidence
                ]
            } if text_result else None,  # FIXED: Added space before 'if'

            "recommendation": (
                "HIGH RISK: Flag for manual review"
                if risk_level == "HIGH"
                else "MEDIUM RISK: Needs verification"
                if risk_level == "MEDIUM"
                else "LOW RISK: Likely authentic"
            )
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# ------------------------------------------------------------------
# HELPER FUNCTIONS FOR FIELD MAPPING
# ------------------------------------------------------------------
def _extract_attention_highlights(evidence, text):
    """Extract keywords from evidence for attention visualization"""
    highlights = []
    
    # Convert evidence to strings
    evidence_text = ' '.join([
        str(e) if isinstance(e, str) else str(e) 
        for e in evidence
    ]).lower()
    
    # Extract key patterns
    patterns = {
        'BREAKING': 0.9,
        'breaking': 0.9,
        'secret': 0.85,
        'shocking': 0.85,
        'patterns': 0.8,
        'indicators': 0.75,
        'sources': 0.7,
        'credibility': 0.7,
        'emotion': 0.6,
        'manipulation': 0.6,
        'suspicious': 0.8,
        'fake': 0.85,
        'uncertain': 0.5
    }
    
    for word, score in patterns.items():
        if word.lower() in evidence_text:
            highlights.append({
                "word": word.upper() if word.isupper() else word.capitalize(),
                "score": score
            })
    
    # Limit to 8 highlights
    return highlights[:8] if highlights else [{"word": "Analyzing", "score": 0.5}]

def _extract_highlighted_phrases(evidence):
    """Extract phrases to highlight in the text"""
    phrases = ['breaking', 'urgent', 'shocking', 'secret', 'deal', 'alert', 'exclusive']
    
    # Extract from evidence
    for e in evidence:
        e_str = str(e).lower()
        if 'breaking' in e_str:
            phrases.append('breaking')
        if 'secret' in e_str:
            phrases.append('secret')
        if 'deal' in e_str:
            phrases.append('deal')
    
    return list(set(phrases))  # Remove duplicates
    
# ------------------------------------------------------------------
# REVERSE IMAGE SEARCH
# ------------------------------------------------------------------
@app.route('/api/reverse-image-search', methods=['POST'])
def reverse_image_search():
    try:
        from serpapi import GoogleSearch

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        serpapi_key = request.form.get('serpapi_key')
        imgbb_key = request.form.get('imgbb_key')

        if not serpapi_key or not imgbb_key:
            return jsonify({'error': 'API keys required'}), 400

        filename = secure_filename(image_file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(path)

        # Upload to ImgBB
        with open(path, "rb") as f:
            upload = requests.post(
                "https://api.imgbb.com/1/upload",
                params={"key": imgbb_key},
                files={"image": f}
            )

        os.remove(path)

        if upload.status_code != 200:
            return jsonify({'error': 'Image upload failed'}), 500

        image_url = upload.json()["data"]["url"]

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": serpapi_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        sources = []
        for item in results.get("visual_matches", [])[:10]:
            sources.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "source": item.get("source")
            })

        return jsonify({
            "status": "success",
            "sources": sources,
            "image_url": image_url
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔥 Server running on http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=True)