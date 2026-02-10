"""
Flask API for Zero-Shot Deepfake & Misinformation Detection
=========================================================

Production-ready REST API with:
- Image detection → EnhancedZeroShotVisualDetectorV2
- Text detection  → ZeroShotDeepfakeDetectionSystem
- Reverse Image Search (SerpAPI + ImgBB)
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

from Phase_2.zero_shot_visual_detector import EnhancedZeroShotVisualDetectorV2
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
print("\n🚀 Loading detection engines...")

try:
    visual_detector = EnhancedZeroShotVisualDetectorV2()
    print("✅ Visual detector loaded")
except Exception as e:
    print(f"❌ Visual detector failed: {e}")
    visual_detector = None

try:
    detector = ZeroShotDetectionSystem()
    print("✅ Text / multimodal detector loaded")
except Exception as e:
    print(f"❌ Text detector failed: {e}")
    detector = None

def to_native(value):
    """
    Convert numpy / torch scalars to native Python types
    """
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value
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
        "status": "healthy" if detector and visual_detector else "unhealthy",
        "visual_loaded": visual_detector is not None,
        "text_loaded": detector is not None
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
# TEXT ONLY (TEXT DETECTOR)
# ------------------------------------------------------------------
@app.route("/api/detect-text", methods=["POST"])
def detect_text_only():
    if detector is None:
        return jsonify({"error": "Text detector unavailable"}), 503

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = detector.detect(text=text)

        return jsonify({
            "verdict": result["verdict"],
            "authenticity_score": round(result["score"], 3),
            "misinformation_score": authenticity_to_misinformation(result["score"]),
            "confidence": round(result.get("confidence", 0.5), 2),
            "explanation": result.get("explanation", "")
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# FULL MULTIMODAL (unchanged behavior)
# ------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def detect():
    if detector is None:
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

        result = detector.detect(
            text=text,
            image_path=image_path,
            return_detailed=True
        )

        for f in uploaded:
            try:
                os.remove(f)
            except:
                pass

        fake_prob = float(result.get("fake_probability", 0.5))

        return jsonify({
            "verdict": result["verdict"],
            "misinformation_score": round(fake_prob * 100.0, 1),
            "confidence": float(result.get("confidence", 0.0)),
            "risk_level": result.get("risk_level", "MEDIUM"),
            "explanation": result.get("explanation", ""),
            "agent_scores": result.get("agent_scores", {})
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze_frontend():
    """
    Frontend dashboard endpoint
    Higher score = higher misinformation risk
    """
    if detector is None and visual_detector is None:
        return jsonify({
            "error": "Detection engines not available",
            "status": "error"
        }), 503

    try:
        text = request.form.get("text", "").strip()
        file = request.files.get("file")

        visual_result = None
        text_result = None
        uploaded_files = []

        # ---------------- IMAGE ----------------
        if file and allowed_file(file.filename, "image"):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            uploaded_files.append(path)

            image = Image.open(path).convert("RGB")
            visual_result = visual_detector.analyze_from_image(image)

        # ---------------- TEXT ----------------
        if text:
            text_result = detector.detect(text=text)

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
            scores.append(1.0 - visual_result.score)

        if text_result:
            # text_result is a DetectionResult object
            scores.append(text_result.fake_probability)

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
            text_risk = to_native(text_result.fake_probability) * 100.0
            evidence_chain.append({
                "type": "textual",
                "score": round(text_risk, 1),
                "weight": 0.4,
                "reason": f"Text analysis verdict: {text_result.verdict}"
            })
        # ---------------- RESPONSE ----------------
        response = {
            "status": "success",
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "evidence_chain": evidence_chain,
            "visual": {
                "misinformation_score": round(
                    to_native(1.0 - visual_result.score) * 100.0, 1
                ),
                "confidence": round(to_native(visual_result.confidence), 2),
                "verdict": visual_result.verdict,
                "evidence": [
                    to_native(e) for e in visual_result.evidence
                ]
            } if visual_result else None,

            "textual": {
                "misinformation_score": round(to_native(text_result.fake_probability) * 100.0, 1),
                "confidence": round(to_native(text_result.confidence), 2),
                "verdict": text_result.verdict,
                "risk_level": text_result.risk_level,
                "explanation": text_result.explanation
            }if text_result else None,

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
# REVERSE IMAGE SEARCH (UNCHANGED)
# ------------------------------------------------------------------
@app.route('/api/reverse-image-search', methods=['POST'])
def reverse_image_search():
    try:
        from config import Config
        from serpapi import GoogleSearch

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        serpapi_key = request.form.get('serpapi_key') or Config.SERPAPI_KEY
        imgbb_key = Config.IMGBB_API_KEY

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
    app.run(host="0.0.0.0", port=5001, debug=False)