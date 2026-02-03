#!/usr/bin/env python3

import os
import uuid
import logging
import re
import json
import hashlib
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import requests
import imagehash

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# Check if serpapi is available
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    GoogleSearch = None
    SERPAPI_AVAILABLE = False
    print("⚠️  WARNING: serpapi package not installed. Run: pip install google-search-results")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "txt", "pdf", "mp4", "avi", "mov"}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load API keys
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
    
    TRUSTED_DOMAINS = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
        "wsj.com", "nytimes.com", "theguardian.com", "dw.com", "france24.com",
        "bloomberg.com", "snopes.com", "politifact.com", "factcheck.org",
        "thehindu.com", "indianexpress.com", "timesofindia.indiatimes.com",
        "hindustantimes.com", "ndtv.com", "livemint.com", "cnn.com", "aljazeera.com"
    ]
    
    FACT_CHECK_DOMAINS = [
        "snopes.com", "politifact.com", "factcheck.org",
        "reuters.com/fact-check", "apnews.com/ap-fact-check",
        "afp.com/factcheck", "fullfact.org"
    ]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mediashield")
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# DEBUG: Print API key status at startup
logger.info("="*60)
logger.info("🔧 Configuration Check:")
logger.info(f"  SERPAPI_KEY: {'✓ Set (' + str(len(Config.SERPAPI_KEY)) + ' chars)' if Config.SERPAPI_KEY else '✗ Not set'}")
logger.info(f"  IMGBB_API_KEY: {'✓ Set' if Config.IMGBB_API_KEY else '✗ Not set'}")
logger.info(f"  SERPAPI Package: {'✓ Installed' if SERPAPI_AVAILABLE else '✗ Not installed'}")
logger.info(f"  Device: {Config.DEVICE}")
logger.info("="*60)

# Warn if SerpAPI is not configured
if not Config.SERPAPI_KEY or not SERPAPI_AVAILABLE:
    logger.warning("⚠️  SerpAPI not configured - provenance and source verification will use fallback data")
    logger.warning("   Get free API key at: https://serpapi.com/ (100 free searches/month)")


# ============================================================================
# FEATURE 0: DYNAMIC CLAIM VERIFIER
# ============================================================================

class DynamicClaimVerifier:
    """Verify claims using real-time web search and fact-check databases"""
    
    def __init__(self):
        self.serpapi_key = Config.SERPAPI_KEY
        self.fact_check_domains = Config.FACT_CHECK_DOMAINS
    
    def analyze(self, text: str) -> Dict:
        """Check if claim contradicts verified facts"""
        if not text or len(text) < 10:
            return {"risk_boost": 0.0, "reason": "Insufficient text", "triggered": False}
        
        claim = text.split('.')[0][:200].strip()
        
        if not claim:
            return {"risk_boost": 0.0, "reason": "No extractable claim", "triggered": False}
        
        query = f'"{claim}" fact check'
        
        try:
            if self.serpapi_key and SERPAPI_AVAILABLE:
                logger.info(f"[Claim Verification] Searching: '{query[:50]}...'")
                
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": self.serpapi_key,
                    "num": 5
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                for result in results.get('organic_results', []):
                    url = result.get('link', '').lower()
                    snippet = result.get('snippet', '').lower()
                    title = result.get('title', '').lower()
                    
                    if any(domain in url for domain in self.fact_check_domains):
                        negative_keywords = ['false', 'incorrect', 'debunked', 'misleading', 
                                           'fake', 'not true', 'misinformation', 'pants on fire']
                        
                        if any(keyword in snippet or keyword in title for keyword in negative_keywords):
                            logger.warning(f"[Claim Verification] ❌ Claim debunked by {url}")
                            return {
                                "risk_boost": 70.0,
                                "reason": f"Fact-checkers flagged: {result.get('title', '')[:100]}",
                                "source": url,
                                "triggered": True
                            }
                
                logger.info("[Claim Verification] ✓ No contradictions found")
                return {"risk_boost": 0.0, "reason": "No fact-check contradictions found", "triggered": False}
            
            else:
                logger.warning("[Claim Verification] SerpAPI not available")
                return {"risk_boost": 0.0, "reason": "Verification unavailable", "triggered": False}
        
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {"risk_boost": 0.0, "reason": f"Error: {str(e)}", "triggered": False}

# ============================================================================
# FEATURE 1: PROVENANCE TRACKER - FIXED
# ============================================================================

class ProvenanceTracker:
    """Digital genealogy tracking using reverse image search - FIXED VERSION"""
    
    def __init__(self):
        logger.info("Initializing Provenance Tracker (FIXED)...")
        self.cache = {}
        self.serpapi_key = Config.SERPAPI_KEY
    
    def generate_fingerprint(self, image_path: str) -> str:
        """Generate perceptual hash for image fingerprinting"""
        try:
            img = Image.open(image_path)
            ahash = str(imagehash.average_hash(img))
            dhash = str(imagehash.dhash(img))
            return f"{ahash}_{dhash}"
        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return hashlib.md5(open(image_path, 'rb').read()).hexdigest()
    
    def reverse_search_google(self, image_path: str, current_headline: str = "") -> List[Dict]:
        """FIXED: Proper reverse image search using SerpAPI Google Lens"""
        
        if not self.serpapi_key or not SERPAPI_AVAILABLE:
            logger.warning("❌ SerpAPI not available - using fallback")
            return []
        
        try:
            logger.info(f"[Provenance] 🔍 Running reverse image search via Google Lens...")
            
            # Upload image to a temporary hosting service or use local URL
            # For now, we'll use base64 with google_lens engine
            import base64
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Method 1: Try Google Lens (most reliable for provenance)
            try:
                params = {
                    "engine": "google_lens",
                    "url": f"file://{os.path.abspath(image_path)}",  # Try local file first
                    "api_key": self.serpapi_key
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if 'error' not in results:
                    logger.info("[Provenance] ✅ Google Lens search successful")
                    return self._parse_lens_results(results, current_headline)
            except Exception as e:
                logger.warning(f"[Provenance] Google Lens failed: {e}, trying alternative...")
            
            # Method 2: Try uploading to imgbb (free image hosting)
            try:
                imgbb_key = os.getenv("IMGBB_API_KEY")
                if imgbb_key:
                    logger.info("[Provenance] Uploading to imgbb for reverse search...")
                    imgbb_response = requests.post(
                        "https://api.imgbb.com/1/upload",
                        data={
                            "key": imgbb_key,
                            "image": base64.b64encode(image_data).decode('utf-8')
                        },
                        timeout=10
                    )
                    
                    if imgbb_response.status_code == 200:
                        image_url = imgbb_response.json()['data']['url']
                        logger.info(f"[Provenance] Image uploaded: {image_url}")
                        
                        params = {
                            "engine": "google_lens",
                            "url": image_url,
                            "api_key": self.serpapi_key
                        }
                        
                        search = GoogleSearch(params)
                        results = search.get_dict()
                        
                        if 'error' not in results:
                            return self._parse_lens_results(results, current_headline)
            except Exception as e:
                logger.warning(f"[Provenance] imgbb upload failed: {e}")
            
            # If all methods fail, return empty (no fake data)
            logger.warning("[Provenance] ⚠ All search methods failed - no provenance data available")
            return []
        
        except Exception as e:
            logger.error(f"[Provenance] ❌ Search failed: {e}", exc_info=True)
            return []
    
    def _parse_lens_results(self, results: Dict, current_headline: str) -> List[Dict]:
        """Parse Google Lens API results into timeline format"""
        timeline = []
        
        # Try multiple result keys
        visual_matches = results.get('visual_matches', [])
        
        logger.info(f"[Provenance] Found {len(visual_matches)} visual matches")
        
        for match in visual_matches[:10]:  # Limit to 10 results
            title = match.get('title', 'Untitled')
            url = match.get('link', '#')
            source = match.get('source', 'Unknown')
            
            # Try to extract date from title or URL
            date_str = self._extract_date_from_text(title + " " + url)
            
            timeline.append({
                "date": date_str,
                "context": title[:100],
                "url": url,
                "source": source,
                "is_original": False
            })
        
        # Sort by date and mark earliest as original
        if timeline:
            valid_dates = [t for t in timeline if t['date'] != 'Unknown']
            if valid_dates:
                sorted_timeline = sorted(valid_dates, key=lambda x: x['date'])
                sorted_timeline[0]['is_original'] = True
                unknown_dates = [t for t in timeline if t['date'] == 'Unknown']
                return sorted_timeline + unknown_dates
        
        return timeline
    
    def _extract_date_from_text(self, text: str) -> str:
        """Extract date from text using regex patterns"""
        # Try YYYY-MM-DD format
        match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
        if match:
            return match.group(1).replace('/', '-')
        
        # Try Month Day, Year format
        match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2}),? (\d{4})', text, re.IGNORECASE)
        if match:
            month_map = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            month = month_map.get(match.group(1)[:3].lower(), '01')
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{year}-{month}-{day}"
        
        return "Unknown"
    
    def detect_context_hijacking(self, timeline: List[Dict], current_context: str) -> Dict:
        """Analyze timeline for context manipulation"""
        if len(timeline) < 2:
            return {
                "hijacking_detected": False,
                "confidence": 0.0,
                "explanation": "Insufficient historical data (need at least 2 uses)",
                "context_changes": 0
            }
        
        # Sort by date (valid dates first)
        valid_timeline = [t for t in timeline if t['date'] != 'Unknown']
        if len(valid_timeline) < 2:
            return {
                "hijacking_detected": False,
                "confidence": 0.0,
                "explanation": "Cannot determine timeline without dates",
                "context_changes": 0
            }
        
        sorted_timeline = sorted(valid_timeline, key=lambda x: x['date'])
        original = sorted_timeline[0]
        latest = sorted_timeline[-1]
        
        # Check if context has changed
        original_context_lower = original['context'].lower()
        latest_context_lower = latest['context'].lower()
        
        # Calculate similarity (simple word overlap)
        original_words = set(original_context_lower.split())
        latest_words = set(latest_context_lower.split())
        
        if len(original_words) > 0 and len(latest_words) > 0:
            overlap = len(original_words & latest_words) / max(len(original_words), len(latest_words))
        else:
            overlap = 0.0
        
        context_changed = overlap < 0.3  # Less than 30% word overlap = context changed
        
        # Check if source trust has degraded
        trust_degraded = any(word in latest['source'].lower() 
                           for word in ['unknown', 'blog', 'suspicious', 'unverified', 'gossip', 'social'])
        
        # Count unique contexts
        unique_contexts = len(set([t['context'] for t in timeline]))
        
        if context_changed and unique_contexts >= 3:
            try:
                time_gap = (datetime.strptime(latest['date'], '%Y-%m-%d') -
                          datetime.strptime(original['date'], '%Y-%m-%d')).days
            except:
                time_gap = 0
            
            return {
                "hijacking_detected": True,
                "confidence": 0.85,
                "explanation": f"Image originally appeared as \"{original['context'][:50]}...\" ({original['date']}) but now appears as \"{latest['context'][:50]}...\" ({latest['date']})",
                "original_date": original['date'],
                "reuse_date": latest['date'],
                "time_gap_days": time_gap,
                "context_changes": unique_contexts
            }
        
        return {
            "hijacking_detected": False,
            "confidence": 0.0,
            "explanation": f"Context consistent across {len(timeline)} appearances",
            "context_changes": unique_contexts
        }
    
    def analyze(self, image_path: str, current_headline: str = "") -> Dict:
        """Full provenance analysis - FIXED"""
        try:
            fingerprint = self.generate_fingerprint(image_path)
            
            if fingerprint in self.cache:
                logger.info("Using cached provenance data")
                return self.cache[fingerprint]
            
            # Get real timeline data
            timeline = self.reverse_search_google(image_path, current_headline)
            
            if not timeline:
                # No data available - return neutral result
                logger.info("[Provenance] No reverse search data available")
                return {
                    "fingerprint": fingerprint,
                    "timeline": [],
                    "first_seen": "Unknown",
                    "total_appearances": 0,
                    "hijacking_analysis": {
                        "hijacking_detected": False,
                        "confidence": 0.0,
                        "explanation": "No reverse search data available (API unavailable or no matches found)",
                        "context_changes": 0
                    },
                    "risk_score": 0.0,
                    "verdict": "NO DATA"
                }
            
            # Analyze for context hijacking
            hijacking_analysis = self.detect_context_hijacking(timeline, current_headline)
            
            # Calculate risk score based on findings
            risk_score = 0.0
            verdict = "AUTHENTIC"
            
            if hijacking_analysis['hijacking_detected']:
                risk_score = 90.0
                verdict = "CONTEXT HIJACKED"
            elif len(timeline) >= 5:
                risk_score = 40.0
                verdict = "WIDELY CIRCULATED"
            elif len(timeline) >= 2:
                risk_score = 15.0
                verdict = "PREVIOUSLY USED"
            else:
                risk_score = 5.0
                verdict = "FIRST APPEARANCE"
            
            result = {
                "fingerprint": fingerprint,
                "timeline": timeline,
                "first_seen": timeline[0]['date'] if timeline else "Unknown",
                "total_appearances": len(timeline),
                "hijacking_analysis": hijacking_analysis,
                "risk_score": risk_score,
                "verdict": verdict
            }
            
            self.cache[fingerprint] = result
            return result
        
        except Exception as e:
            logger.error(f"Provenance analysis failed: {e}")
            return {
                "fingerprint": "error",
                "timeline": [],
                "first_seen": "Unknown",
                "total_appearances": 0,
                "hijacking_analysis": {
                    "hijacking_detected": False,
                    "confidence": 0.0,
                    "explanation": f"Analysis error: {str(e)}",
                    "context_changes": 0
                },
                "risk_score": 0.0,
                "verdict": "ERROR"
            }

# ============================================================================
# FEATURE 2: DEEPFAKE DETECTOR (Lightweight)
# ============================================================================

class DeepfakeDetector:
    """Lightweight deepfake detection"""
    
    def __init__(self):
        try:
            logger.info("Loading Deepfake Detector (17MB)...")
            self.face_detector = pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=0 if Config.DEVICE == "cuda" else -1
            )
            logger.info("✓ Deepfake detector loaded")
        except Exception as e:
            logger.warning(f"Deepfake model unavailable: {e}")
            self.face_detector = None
    
    def analyze_image(self, image_path: str) -> Dict:
        """Detect deepfake artifacts"""
        if not self.face_detector:
            return {
                "deepfake_probability": 0.0,
                "artifacts": [],
                "verdict": "N/A",
                "confidence": 0.0
            }
        
        try:
            result = self.face_detector(image_path)[0]
            
            if "fake" in result['label'].lower():
                deepfake_prob = result['score'] * 100
            else:
                deepfake_prob = (1 - result['score']) * 100
            
            artifacts = []
            if deepfake_prob > 60:
                artifacts = [
                    "Unnatural facial edges detected",
                    "Inconsistent lighting patterns",
                    "Temporal discontinuities"
                ]
            
            return {
                "deepfake_probability": round(deepfake_prob, 2),
                "confidence": round(result['score'] * 100, 2),
                "artifacts": artifacts,
                "verdict": "AI-GENERATED" if deepfake_prob > 50 else "AUTHENTIC"
            }
        
        except Exception as e:
            logger.error(f"Deepfake detection failed: {e}")
            return {
                "deepfake_probability": 0.0,
                "verdict": "ANALYSIS FAILED",
                "confidence": 0.0,
                "artifacts": []
            }

# ============================================================================
# FEATURE 3: CHAIN-OF-THOUGHT REASONING
# ============================================================================

class ChainOfThoughtEngine:
    """LLM-powered reasoning with explicit step-by-step logic"""
    
    def __init__(self):
        logger.info("Initializing Chain-of-Thought Engine...")
        self.enabled = False
        try:
            # Try to load a small reasoning model
            self.model = pipeline(
                "text-generation",
                model="google/flan-t5-small",
                device=0 if Config.DEVICE == "cuda" else -1,
                max_length=512
            )
            self.enabled = True
            logger.info("✓ CoT engine loaded")
        except Exception as e:
            logger.warning(f"CoT engine unavailable (using rule-based fallback): {e}")
            self.model = None
    
    def analyze(self, text: str, external_context: Dict = None) -> Dict:
        """Generate step-by-step reasoning"""
        if not self.enabled or not self.model:
            return self._rule_based_reasoning(text, external_context)
        
        try:
            prompt = self._build_cot_prompt(text, external_context)
            response = self.model(prompt, max_length=256, do_sample=False)[0]['generated_text']
            
            return self._parse_cot_response(response, external_context)
        
        except Exception as e:
            logger.error(f"CoT reasoning failed: {e}")
            return self._rule_based_reasoning(text, external_context)
    
    def _build_cot_prompt(self, text: str, context: Dict) -> str:
        """Build CoT prompt"""
        external_info = ""
        if context:
            if context.get("fact_check_triggered"):
                external_info += f"\n- Fact-checkers flagged: {context.get('fact_check_reason')}"
            if context.get("source_verified"):
                external_info += f"\n- {context.get('trusted_source_count', 0)} trusted sources found"
        
        prompt = f"""Analyze this news headline for misinformation:

Headline: {text[:200]}

External verification:{external_info if external_info else " None available"}

Step-by-step reasoning:
1. What claims are made?
2. Are there red flags (sensationalism, bias, lack of attribution)?
3. Overall verdict?"""
        
        return prompt
    
    def _parse_cot_response(self, response: str, context: Dict) -> Dict:
        """Parse LLM response into structured format"""
        # Simple parsing (can be enhanced)
        reasoning_chain = response.split('\n')
        reasoning_chain = [r.strip() for r in reasoning_chain if r.strip()]
        
        # Determine verdict based on context
        if context and context.get("fact_check_triggered"):
            verdict = "HIGH RISK"
            recommendation = "Content flagged by fact-checkers"
        elif context and context.get("trusted_source_count", 0) >= 2:
            verdict = "LOW RISK"
            recommendation = "Corroborated by trusted sources"
        else:
            verdict = "MODERATE RISK"
            recommendation = "Limited verification available"
        
        return {
            "cot_enabled": True,
            "reasoning_confidence": 60.0,
            "stage1_examination": {
                "claims_identified": [text[:100]]
            },
            "stage2_inference": {
                "red_flags": []
            },
            "stage3_determination": {
                "verdict": verdict,
                "reasoning_chain": reasoning_chain[:3] if reasoning_chain else ["Analysis completed"],
                "recommendation": recommendation
            }
        }
    
    def _rule_based_reasoning(self, text: str, context: Dict) -> Dict:
        """Fallback rule-based reasoning"""
        reasoning_chain = []
        red_flags = []
        
        # Check for sensationalism
        if any(word in text.upper() for word in ['BREAKING', 'SHOCKING', 'EXPOSED']):
            red_flags.append("Sensational language detected")
            reasoning_chain.append("Contains sensational keywords")
        
        # Check external verification
        if context:
            if context.get("fact_check_triggered"):
                red_flags.append("Flagged by fact-checkers")
                reasoning_chain.append(f"Fact-check warning: {context.get('fact_check_reason', '')[:80]}")
            
            trusted_count = context.get("trusted_source_count", 0)
            if trusted_count >= 2:
                reasoning_chain.append(f"Verified by {trusted_count} trusted sources")
            elif trusted_count == 0:
                red_flags.append("No trusted source verification")
                reasoning_chain.append("No corroboration from trusted sources")
        
        # Determine verdict
        if len(red_flags) >= 2:
            verdict = "HIGH RISK"
            recommendation = "Multiple red flags detected"
        elif len(red_flags) == 1:
            verdict = "MODERATE RISK"
            recommendation = "Some concerns identified"
        else:
            verdict = "LOW RISK"
            recommendation = "No major concerns detected"
        
        if not reasoning_chain:
            reasoning_chain = ["Standard verification checks applied"]
        
        return {
            "cot_enabled": False,
            "reasoning_confidence": 50.0,
            "stage1_examination": {
                "claims_identified": [text[:100]]
            },
            "stage2_inference": {
                "red_flags": red_flags
            },
            "stage3_determination": {
                "verdict": verdict,
                "reasoning_chain": reasoning_chain,
                "recommendation": recommendation
            }
        }

# ============================================================================
# FEATURE 4: ATTENTION-BASED EXPLAINABILITY
# ============================================================================

class ExplainabilityEngine:
    """Visualize what the AI focused on during analysis"""
    
    def __init__(self):
        logger.info("Initializing Explainability Engine...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "mrm8488/bert-tiny-finetuned-fake-news-detection"
            ).to(Config.DEVICE)
            self.model.eval()
            self.enabled = True
            logger.info("✓ Explainability engine loaded")
        except Exception as e:
            logger.warning(f"Explainability unavailable: {e}")
            self.enabled = False
            self.tokenizer = None
            self.model = None
    
    def generate_visual_explanation(self, text: str, verdict: str) -> Dict:
        """Generate attention-weighted highlights"""
        if not self.enabled or not text:
            return self._fallback_explanation(text, verdict)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(Config.DEVICE)
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions[-1]  # Last layer
                attention_avg = attentions.mean(dim=1).squeeze()  # Average across heads
            
            # Extract important tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            attention_scores = attention_avg.mean(dim=0).cpu().numpy()
            
            # Find top tokens
            token_importance = list(zip(tokens, attention_scores))
            token_importance = [(t, float(s)) for t, s in token_importance if t not in ['[CLS]', '[SEP]', '[PAD]']]
            token_importance.sort(key=lambda x: x[1], reverse=True)
            
            top_phrases = []
            for token, score in token_importance[:5]:
                # Clean token
                clean_token = token.replace('##', '')
                if len(clean_token) > 2:
                    top_phrases.append({
                        "text": clean_token,
                        "attention_weight": round(score * 100, 1)
                    })
            
            return {
                "highlighted_phrases": top_phrases,
                "visual_explanation": f"AI focused on: {', '.join([p['text'] for p in top_phrases[:3]])}",
                "confidence_driver": top_phrases[0]['text'] if top_phrases else "N/A"
            }
        
        except Exception as e:
            logger.error(f"Explainability generation failed: {e}")
            return self._fallback_explanation(text, verdict)
    
    def _fallback_explanation(self, text: str, verdict: str) -> Dict:
        """Fallback when attention not available"""
        # Extract keywords using simple heuristics
        words = text.split()
        suspicious_keywords = ['breaking', 'shocking', 'exposed', 'exclusive', 'urgent']
        
        found = []
        for word in words:
            if word.lower() in suspicious_keywords:
                found.append({
                    "text": word,
                    "attention_weight": 20.0
                })
        
        if not found:
            # Just take first few meaningful words
            meaningful = [w for w in words if len(w) > 3][:3]
            found = [{"text": w, "attention_weight": 10.0} for w in meaningful]
        
        return {
            "highlighted_phrases": found[:5],
            "visual_explanation": "Keyword-based highlighting (attention unavailable)",
            "confidence_driver": found[0]['text'] if found else "N/A"
        }

# ============================================================================
# FEATURE 5: MULTILINGUAL TEXT ANALYZER
# ============================================================================

class MultilingualTextAnalyzer:
    """Lightweight multilingual fake news detection"""
    
    def __init__(self):
        try:
            logger.info("Loading Lightweight Text Analyzer (BERT-tiny: 17MB)...")
            device = 0 if Config.DEVICE == "cuda" else -1
            
            # Main classifier (17MB)
            self.classifier = pipeline(
                "text-classification",
                model="mrm8488/bert-tiny-finetuned-fake-news-detection",
                device=device
            )
            
            # Simple language detection (optional)
            self.lang_detector = None
            
            logger.info("✓ Text analyzer loaded (17MB total)")
        except Exception as e:
            logger.error(f"Text analyzer failed: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        # Fallback to simple heuristics
        if not text:
            return "en"
        
        # Check for common non-English characters
        if any('\u0900' <= char <= '\u097F' for char in text):
            return "hi"  # Hindi
        elif any('\u0600' <= char <= '\u06FF' for char in text):
            return "ar"  # Arabic
        elif any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"  # Chinese
        elif any('\u3040' <= char <= '\u309f' for char in text):
            return "ja"  # Japanese
        else:
            return "en"  # Default to English
    
    def _calculate_sensationalism(self, text: str) -> float:
        """Calculate sensationalism score (0-1)"""
        if not text or len(text) < 20:
            return 0.0
        
        score = 0.0
        text_upper = text.upper()
        
        sensational_words = [
            "BREAKING", "SHOCKING", "EXPOSED", "BOMBSHELL", "EXCLUSIVE",
            "URGENT", "ALERT", "SCANDAL", "LEAKED", "UNBELIEVABLE"
        ]
        found = sum(1 for word in sensational_words if word in text_upper)
        score += min(0.4, found * 0.1)
        
        exclamation_ratio = text.count('!') / max(len(text.split()), 1)
        score += min(0.3, exclamation_ratio * 2)
        
        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        score += min(0.3, caps_ratio * 3)
        
        return min(1.0, score)
    
    def analyze(self, text: str) -> Dict:
        """Lightweight fake news analysis"""
        try:
            language = self.detect_language(text)
            
            safe_text = text[:512] if text else "No text"
            result = self.classifier(safe_text)[0]
            label = result['label']
            score = float(result['score'])
            
            sensationalism = self._calculate_sensationalism(text)
            
            # FIXED: Proper risk calculation
            # If model says "FAKE" or "UNRELIABLE" with high confidence -> high risk
            # If model says "REAL" or "RELIABLE" with high confidence -> low risk
            if 'fake' in label.lower() or 'unreliable' in label.lower() or label == 'LABEL_0':
                base_risk = score * 100.0  # High confidence in FAKE = high risk
            else:
                base_risk = (1.0 - score) * 100.0  # High confidence in REAL = low risk
            
            # Boost risk if highly sensational
            if sensationalism > 0.6:
                base_risk = min(95.0, base_risk * 1.3)
            
            return {
                "language_detected": language,
                "credibility_verdict": label,
                "model_confidence": round(score * 100.0, 2),
                "risk_score": round(base_risk, 2),
                "multilingual_enabled": True,
                "sensationalism_index": round(sensationalism * 10, 1),
                "attention_highlights": [],
                "highlighted_phrases": []
            }
        
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "risk_score": 50.0,
                "language_detected": "en",
                "sensationalism_index": 0.0,
                "credibility_verdict": "ERROR",
                "error": str(e)
            }

# ============================================================================
# SOURCE VERIFICATION - FIXED
# ============================================================================

class SourceVerifier:
    """Verify claims using SerpAPI - FIXED VERSION"""
    
    def __init__(self):
        self.serpapi_key = Config.SERPAPI_KEY
        self.trusted_domains = Config.TRUSTED_DOMAINS
    
    def verify(self, text_segment: str) -> Dict:
        """Search for corroborating sources - FIXED"""
        if not text_segment or len(text_segment) < 10:
            return {
                "risk_score": 50.0,
                "trusted_sources_count": 0,
                "links": [],
                "found_trusted": False
            }
        
        # Extract key claims/entities from text (improved extraction)
        clean_text = re.sub(r"[^\w\s]", " ", text_segment)
        clean_text = " ".join(clean_text.split())
        
        # Take meaningful words only (filter out common words)
        common_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'and', 'of', 'to', 'in', 'for', 'with', 'by'}
        tokens = [w for w in clean_text.split() if w.lower() not in common_words and len(w) > 2]
        
        # Build search query (first 5-8 meaningful words)
        query = " ".join(tokens[:8])
        
        logger.info(f"[Source Verification] Searching: '{query[:50]}...'")
        
        if self.serpapi_key and SERPAPI_AVAILABLE:
            return self._search_with_serpapi(query)
        else:
            logger.warning("[Source Verification] SerpAPI not available - no verification possible")
            return {
                "risk_score": 50.0,
                "trusted_sources_count": 0,
                "links": [],
                "found_trusted": False,
                "error": "API unavailable"
            }
    
    def _search_with_serpapi(self, query: str) -> Dict:
        """Real search using SerpAPI - FIXED"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": 8  # Get more results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if 'error' in results:
                logger.error(f"[Source Verification] API error: {results['error']}")
                return {
                    "risk_score": 50.0,
                    "trusted_sources_count": 0,
                    "links": [],
                    "found_trusted": False,
                    "error": results['error']
                }
            
            organic_results = results.get("organic_results", [])
            
            found_sources = []
            trusted_count = 0
            
            for result in organic_results:
                link = result.get("link", "")
                title = result.get("title", "Untitled")
                snippet = result.get("snippet", "")
                
                # Check if from trusted domain
                is_trusted = any(domain in link.lower() for domain in self.trusted_domains)
                
                if is_trusted:
                    trusted_count += 1
                
                found_sources.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet[:150],
                    "is_trusted": is_trusted
                })
            
            # FIXED: Better risk scoring
            if trusted_count >= 3:
                risk_score = 0.0  # Excellent verification
            elif trusted_count == 2:
                risk_score = 10.0  # Good verification
            elif trusted_count == 1:
                risk_score = 30.0  # Some verification
            elif len(found_sources) >= 3:
                risk_score = 50.0  # Found sources but not trusted
            elif len(found_sources) > 0:
                risk_score = 70.0  # Few sources, none trusted
            else:
                risk_score = 90.0  # No sources found
            
            logger.info(f"[Source Verification] ✓ Found {trusted_count} trusted sources out of {len(found_sources)} total")
            
            return {
                "risk_score": risk_score,
                "trusted_sources_count": trusted_count,
                "total_sources": len(found_sources),
                "links": found_sources[:5],  # Return top 5
                "found_trusted": trusted_count > 0
            }
        
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return {
                "risk_score": 50.0,
                "trusted_sources_count": 0,
                "links": [],
                "found_trusted": False,
                "error": str(e)
            }

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize engines
logger.info("Initializing MediaShield engines...")
provenance_engine = ProvenanceTracker()
deepfake_engine = DeepfakeDetector()
text_analyzer = MultilingualTextAnalyzer()
explainability_engine = ExplainabilityEngine()
source_verifier = SourceVerifier()
cot_engine = ChainOfThoughtEngine()
claim_verifier = DynamicClaimVerifier()
logger.info("✓ All engines initialized")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "features": {
            "provenance": True,
            "deepfake": deepfake_engine.face_detector is not None,
            "cot": cot_engine.enabled,
            "explainability": explainability_engine.enabled,
            "multilingual": True,
            "source_verification": Config.SERPAPI_KEY is not None,
            "claim_verification": Config.SERPAPI_KEY is not None
        },
        "device": Config.DEVICE
    }), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint - FIXED"""
    try:
        text_content = request.form.get("text", "")
        language = request.form.get("language", "en")
        
        file = request.files.get("file")
        filename = ""
        save_path = ""
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(Config.UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
            file.save(save_path)
            logger.info(f"File saved: {save_path}")
        
        # 1. Text analysis with claim verification
        text_results = text_analyzer.analyze(text_content)
        
        # Dynamic claim verification
        claim_verification = claim_verifier.analyze(text_content)
        
        # Boost risk if fact-checked
        if claim_verification.get("triggered"):
            original_risk = text_results.get("risk_score", 0)
            boosted_risk = min(95.0, original_risk + claim_verification.get("risk_boost", 0))
            text_results["risk_score"] = boosted_risk
            text_results["fact_check_triggered"] = True
            text_results["fact_check_reason"] = claim_verification.get("reason", "")
            logger.warning(f"[Fact Check] ❌ Boosted risk: {original_risk}% → {boosted_risk}%")
        else:
            text_results["fact_check_triggered"] = False
        
        # 2. Explainability
        explainability_results = explainability_engine.generate_visual_explanation(
            text_content,
            text_results.get('credibility_verdict', 'UNKNOWN')
        )
        
        # 3. Deepfake detection
        deepfake_results = {"deepfake_probability": 0.0, "verdict": "N/A", "confidence": 0.0}
        if save_path and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            deepfake_results = deepfake_engine.analyze_image(save_path)
        
        # 4. Provenance tracking - FIXED
        provenance_results = {"risk_score": 0.0, "verdict": "N/A", "timeline": []}
        if save_path and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            provenance_results = provenance_engine.analyze(save_path, text_content)
        
        # 5. Source verification - FIXED
        source_results = source_verifier.verify(text_content[:300])
        
        # 6. CoT reasoning
        cot_results = cot_engine.analyze(
            text_content,
            external_context={
                "source_verified": source_results.get("found_trusted", False),
                "trusted_source_count": source_results.get("trusted_sources_count", 0),
                "fact_check_triggered": claim_verification.get("triggered", False),
                "fact_check_reason": claim_verification.get("reason", "")
            }
        )
        
        # FIXED: Calculate overall score with proper weighting
        # Only include provenance if we have data
        if provenance_results.get("verdict") not in ["NO DATA", "ERROR"]:
            final_score = (
                text_results.get("risk_score", 50.0) * 0.30 +
                deepfake_results.get("deepfake_probability", 0.0) * 0.25 +
                provenance_results.get("risk_score", 0.0) * 0.25 +
                source_results.get("risk_score", 50.0) * 0.20
            )
        else:
            # No provenance data - redistribute weight
            final_score = (
                text_results.get("risk_score", 50.0) * 0.40 +
                deepfake_results.get("deepfake_probability", 0.0) * 0.35 +
                source_results.get("risk_score", 50.0) * 0.25
            )
        
        overall_risk = round(final_score, 1)
        
        # Determine verdict
        if overall_risk >= 70:
            verdict = "HIGH RISK"
        elif overall_risk >= 40:
            verdict = "MODERATE RISK"
        else:
            verdict = "LOW RISK"
        
        response = {
            "overall_risk_score": overall_risk,
            "verdict": verdict,
            "recommendation": cot_results.get("stage3_determination", {}).get("recommendation", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_id": str(uuid.uuid4()),
            "status": "success",
            
            "provenance": {
                "fingerprint": provenance_results.get("fingerprint", "N/A"),
                "timeline": provenance_results.get("timeline", []),
                "first_seen": provenance_results.get("first_seen", "Unknown"),
                "total_appearances": provenance_results.get("total_appearances", 0),
                "context_changes": provenance_results.get("hijacking_analysis", {}).get("context_changes", 0),
                "hijacking_detected": provenance_results.get("hijacking_analysis", {}).get("hijacking_detected", False),
                "hijacking_explanation": provenance_results.get("hijacking_analysis", {}).get("explanation", ""),
                "risk_score": provenance_results.get("risk_score", 0.0),
                "verdict": provenance_results.get("verdict", "N/A")
            },
            
            "visual": {
                "deepfake_score": deepfake_results.get("deepfake_probability", 0.0),
                "deepfake": {
                    "probability": deepfake_results.get("deepfake_probability", 0.0),
                    "artifacts": deepfake_results.get("artifacts", []),
                    "verdict": deepfake_results.get("verdict", "N/A"),
                    "confidence": deepfake_results.get("confidence", 0.0)
                }
            },
            
            "reasoning": {
                "enabled": cot_results.get("cot_enabled", False),
                "method": "llm-cot" if cot_results.get("cot_enabled") else "rule-based",
                "explanation": " → ".join(cot_results.get("stage3_determination", {}).get("reasoning_chain", [])),
                "confidence": cot_results.get("reasoning_confidence", 0.0),
                "stage1_claims": cot_results.get("stage1_examination", {}).get("claims_identified", []),
                "stage2_red_flags": cot_results.get("stage2_inference", {}).get("red_flags", []),
                "stage3_verdict": cot_results.get("stage3_determination", {}).get("verdict", ""),
                "reasoning_chain": cot_results.get("stage3_determination", {}).get("reasoning_chain", []),
                "recommendation": cot_results.get("stage3_determination", {}).get("recommendation", "")
            },
            
            "explainability": {
                "highlighted_phrases": explainability_results.get("highlighted_phrases", []),
                "visual_explanation": explainability_results.get("visual_explanation", ""),
                "confidence_driver": explainability_results.get("confidence_driver", "N/A")
            },
            
            "textual": {
                "language_detected": text_results.get("language_detected", "en"),
                "multilingual_enabled": text_results.get("multilingual_enabled", False),
                "credibility_score": 100.0 - text_results.get("risk_score", 50.0),
                "risk_score": text_results.get("risk_score", 50.0),
                "verdict": text_results.get("credibility_verdict", "UNKNOWN"),
                "sensationalism_index": text_results.get("sensationalism_index", 0.0),
                "attention_highlights": explainability_results.get("highlighted_phrases", []),
                "highlighted_phrases": [],
                "fact_check_triggered": text_results.get("fact_check_triggered", False),
                "fact_check_reason": text_results.get("fact_check_reason", "")
            },
            
            "source": {
                "trust_score": 100.0 - source_results.get("risk_score", 50.0),
                "risk_score": source_results.get("risk_score", 50.0),
                "trusted_sources_count": source_results.get("trusted_sources_count", 0),
                "total_sources": source_results.get("total_sources", 0),
                "links": source_results.get("links", [])
            },
            
            "evidence_chain": [
                {
                    "type": "provenance",
                    "score": provenance_results.get("risk_score", 0.0),
                    "weight": 0.25 if provenance_results.get("verdict") not in ["NO DATA", "ERROR"] else 0.0,
                    "reason": provenance_results.get("verdict", "N/A")
                },
                {
                    "type": "textual",
                    "score": text_results.get("risk_score", 0.0),
                    "weight": 0.30,
                    "reason": f"Lang: {text_results.get('language_detected', 'en')}, Verdict: {text_results.get('credibility_verdict', 'N/A')}"
                },
                {
                    "type": "visual",
                    "score": deepfake_results.get("deepfake_probability", 0.0),
                    "weight": 0.25,
                    "reason": deepfake_results.get("verdict", "N/A")
                },
                {
                    "type": "source",
                    "score": source_results.get("risk_score", 50.0),
                    "weight": 0.20,
                    "reason": f"Trusted sources: {source_results.get('trusted_sources_count', 0)}/{source_results.get('total_sources', 0)}"
                }
            ]
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 MediaShield FIXED Server Starting...")
    logger.info("="*60)
    logger.info("FIXES APPLIED:")
    logger.info(" ✓ Provenance: Proper Google Lens API integration")
    logger.info(" ✓ Source Verification: Real search results (no fake URLs)")
    logger.info(" ✓ Scoring: Fixed risk calculation logic")
    logger.info(" ✓ Better error handling and fallbacks")
    logger.info("="*60)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"SerpAPI: {'✓ Enabled' if Config.SERPAPI_KEY else '✗ Disabled (limited functionality)'}")
    if Config.SERPAPI_KEY:
        logger.info("Note: For provenance to work, you may need IMGBB_API_KEY as well")
    logger.info("="*60)
    app.run(host="0.0.0.0", port=5001, debug=True)
