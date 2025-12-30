# Multimodal_Deepfake

# 1. Clone or download project
git clone <repo> && cd mediashield

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) SerpAPI key for source verification
echo "SERPAPI_KEY=your_key_here" > .env

# 5. Start backend
python app.py

# 6. Open frontend
python -m http.server 8000  # Terminal 2
# Visit: http://localhost:8000

pip install flask flask-cors python-dotenv pillow transformers torch accelerate google-search-results pymupdf reportlab werkzeug
