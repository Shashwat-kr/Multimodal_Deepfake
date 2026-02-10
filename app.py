"""
Flask API for Zero-Shot Deepfake Detection
==========================================

Production-ready REST API with CORS support.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from Phase_2.zero_shot_detection_system import ZeroShotDeepfakeDetectionSystem

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'},
    'audio': {'mp3', 'wav', 'ogg', 'flac', 'm4a'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'}
}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detection system (load once)
print("\n🚀 Loading Zero-Shot Deepfake Detection System...")
try:
    detector = ZeroShotDeepfakeDetectionSystem(
        fusion_dim=512,
        num_transformer_layers=3,
        use_large_clip=False,  # Set True for better accuracy (slower)
        verbose=False  # Disable verbose output for API
    )
    print("✅ Detection system loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load detection system: {e}")
    detector = None


def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'][file_type]


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Detection system not initialized'
        }), 503

    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'agents': detector.agent_status if detector else {}
    }), 200

@app.route('/health', methods=['GET'])
def health_simple():
    """
    Frontend health endpoint for script.js (GET /health).
    Returns device + feature flags.
    """
    # Basic device info (you can refine if you expose Config.DEVICE)
    device = "cpu"
    try:
        # If your ZeroShotDeepfakeDetectionSystem exposes a device attribute
        if hasattr(detector, "device"):
            device = str(detector.device)
    except Exception:
        pass

    # Feature flags expected by script.js
    features = {
        "provenance": True,    # set False if not implemented yet
        "deepfake": True,
        "cot": True,           # Chain-of-Thought reasoning
        "multilingual": True,
    }

    status = "healthy" if detector is not None else "unhealthy"

    return jsonify({
        "status": status,
        "device": device,
        "features": features,
    }), 200 if detector is not None else 503


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint

    Accepts:
    - text: Text content (form data or JSON)
    - image: Image file (multipart/form-data)
    - audio: Audio file (multipart/form-data)
    - video: Video file (multipart/form-data)

    Returns:
    - JSON with detection results
    """

    if detector is None:
        return jsonify({
            'error': 'Detection system not initialized',
            'verdict': 'ERROR'
        }), 503

    try:
        # Parse inputs
        text = None
        image_path = None
        audio_path = None
        video_path = None

        # Get text from form data or JSON
        if 'text' in request.form:
            text = request.form['text']
        elif request.is_json and 'text' in request.json:
            text = request.json['text']

        # Handle file uploads
        uploaded_files = []

        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and allowed_file(image_file.filename, 'image'):
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                uploaded_files.append(image_path)

        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and allowed_file(audio_file.filename, 'audio'):
                filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(audio_path)
                uploaded_files.append(audio_path)

        if 'video' in request.files:
            video_file = request.files['video']
            if video_file and allowed_file(video_file.filename, 'video'):
                filename = secure_filename(video_file.filename)
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(video_path)
                uploaded_files.append(video_path)

        # Validate input
        if not any([text, image_path, audio_path, video_path]):
            return jsonify({
                'error': 'No input provided. Please provide text, image, audio, or video.',
                'verdict': 'ERROR'
            }), 400

        # Run detection
        result = detector.detect(
            text=text,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            return_detailed=True
        )

        # Clean up uploaded files
        for filepath in uploaded_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass

        # Format response
        response = {
            'verdict': result['verdict'],
            'fake_probability': float(result['fake_probability']),
            'confidence': float(result['confidence']),
            'risk_level': result['risk_level'],
            'explanation': result.get('explanation', 'No explanation available'),
            'modalities_analyzed': result.get('modalities_analyzed', []),
            'agent_scores': {k: float(v) for k, v in result.get('agent_scores', {}).items()},
            'agent_verdicts': result.get('agent_verdicts', {}),
            'agent_status': result.get('agent_status', {}),
            'errors': result.get('errors', [])
        }

        return jsonify(response), 200

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()

        return jsonify({
            'error': f'Detection failed: {error_msg}',
            'verdict': 'ERROR',
            'fake_probability': 0.5,
            'confidence': 0.0
        }), 500


@app.route('/api/detect-text', methods=['POST'])
def detect_text_only():
    """Text-only detection endpoint"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        text = request.json.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = detector.detect(text=text)

        return jsonify({
            'verdict': result['verdict'],
            'fake_probability': float(result['fake_probability']),
            'confidence': float(result['confidence']),
            'risk_level': result['risk_level'],
            'explanation': result.get('explanation', '')
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-image', methods=['POST'])
def detect_image_only():
    """Image-only detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if not allowed_file(image_file.filename, 'image'):
            return jsonify({'error': 'Invalid image file type'}), 400

        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        result = detector.detect(image_path=image_path)

        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            'verdict': result['verdict'],
            'fake_probability': float(result['fake_probability']),
            'confidence': float(result['confidence']),
            'risk_level': result['risk_level']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_frontend():
    """
    Frontend analysis endpoint for script.js (POST /analyze).
    Accepts:
      - file: image/audio/video file
      - text: optional text context
      - language: optional language code
    Returns a dashboard-friendly JSON structure.
    """
    if detector is None:
        return jsonify({
            "error": "Detection system not initialized",
            "status": "error"
        }), 503

    try:
        text = request.form.get("text", "")
        language = request.form.get("language", "en")

        file = request.files.get("file")
        image_path = None
        audio_path = None
        video_path = None
        uploaded_files = []

        if file:
            filename = secure_filename(file.filename)
            ext = filename.rsplit(".", 1)[-1].lower()
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            uploaded_files.append(save_path)

            # Map to a modality based on extension
            if ext in app.config["ALLOWED_EXTENSIONS"]["image"]:
                image_path = save_path
            elif ext in app.config["ALLOWED_EXTENSIONS"]["audio"]:
                audio_path = save_path
            elif ext in app.config["ALLOWED_EXTENSIONS"]["video"]:
                video_path = save_path

        if not any([text, image_path, audio_path, video_path]):
            return jsonify({
                "error": "No valid input provided",
                "status": "error"
            }), 400

        # Run core detector (detailed)
        result = detector.detect(
            text=text or None,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            return_detailed=True
        )

        # Cleanup temp files
        for p in uploaded_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        # Map core result into dashboard structure
        fake_prob = float(result.get("fake_probability", 0.5))
        risk_level = result.get("risk_level", "MEDIUM")
        confidence = float(result.get("confidence", 0.0))

        # Convert fake probability into an "overall risk score" (0–100)
        overall_risk_score = round(fake_prob * 100.0, 1)

        # Build visual block
        visual_block = {
            "deepfake_score": round(fake_prob * 100.0, 1),
            "confidence": confidence,
            "verdict": result.get("agent_verdicts", {}).get("visual", result.get("verdict", "UNCERTAIN")),
            "artifacts": result.get("visual_artifacts", []),
        }

        # Simple textual block (you can expand later)
        textual_block = {
            "credibility_score": 100.0 - overall_risk_score,
            "sensationalism_index": result.get("sensationalism", 5.0),
            "language_detected": language,
        }

        # Simple source block using web agent score (if present)
        web_score = result.get("agent_scores", {}).get("web", 0.5)
        source_block = {
            "trust_score": round((1.0 - web_score) * 100.0, 1),
            "links": result.get("web_links", []),
        }

        # Provenance placeholder
        provenance_block = {
            "risk_score": 0.0,
            "details": []
        }

        # Reasoning / CoT
        reasoning_block = {
            "cot_summary": result.get("explanation", "No reasoning available yet.")
        }

        # Evidence chain based on agent scores
        agent_scores = result.get("agent_scores", {})
        evidence_chain = []
        if "visual" in agent_scores:
            evidence_chain.append({
                "type": "visual",
                "score": float(agent_scores["visual"]) * 100.0,
                "weight": 0.35,
                "reason": f"Visual agent verdict: {result.get('agent_verdicts', {}).get('visual', 'N/A')}"
            })
        if "consistency" in agent_scores:
            evidence_chain.append({
                "type": "textual",  # or "consistency" if you prefer
                "score": float(agent_scores["consistency"]) * 100.0,
                "weight": 0.30,
                "reason": f"Consistency agent verdict: {result.get('agent_verdicts', {}).get('consistency', 'N/A')}"
            })
        if "web" in agent_scores:
            evidence_chain.append({
                "type": "source",
                "score": float(agent_scores["web"]) * 100.0,
                "weight": 0.25,
                "reason": f"Web agent verdict: {result.get('agent_verdicts', {}).get('web', 'N/A')}"
            })

        recommendation = "LOW RISK: Likely authentic content"
        if risk_level.upper() == "HIGH" or overall_risk_score >= 65:
            recommendation = "HIGH RISK: Flag for manual review"
        elif risk_level.upper() == "MEDIUM":
            recommendation = "MEDIUM RISK: Needs further verification"

        response = {
            "overall_risk_score": overall_risk_score,
            "verdict": result.get("verdict", "UNCERTAIN"),
            "status": "success",
            "deepfake_probability": visual_block["deepfake_score"],
            "detected_language": language,
            "visual": visual_block,
            "textual": textual_block,
            "source": source_block,
            "provenance": provenance_block,
            "reasoning": reasoning_block,
            "evidence_chain": evidence_chain,
            "recommendation": recommendation,
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "status": "error"
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 50MB.',
        'verdict': 'ERROR'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING FLASK API SERVER")
    print("="*80)
    print("\n📡 API Endpoints:")
    print("   • GET  /api/health          - Health check")
    print("   • POST /api/detect          - Full detection (text/image/audio/video)")
    print("   • POST /api/detect-text     - Text-only detection")
    print("   • POST /api/detect-image    - Image-only detection")
    print("\n🌐 Frontend:")
    print("   • GET  /                    - HTML interface")
    print("\n" + "="*80)
    print("\n🔥 Server starting on http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("\n" + "="*80 + "\n")

    # Run Flask app
    app.run(
        host='0.0.0.0',  # Accessible from network
        port=5001,
        debug=False,  # Set True for development
        threaded=True
    )
