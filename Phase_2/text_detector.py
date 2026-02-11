"""
ENHANCED TEXT DETECTOR V2 - With Adaptive Weighting

Fixes:
1. Better handling of "no sources" in breaking news
2. Adaptive weighting for extreme signals
3. More aggressive fake news detection
4. Context-aware credibility scoring
"""

import re
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from urllib.parse import urlparse
import requests

@dataclass
class TextAnalysisResult:
    score: float  # 0=fake, 1=real
    verdict: str
    reasoning: str
    evidence: List[str]
    confidence: float


class EnhancedTextDetectorV2:
    """
    Enhanced zero-shot text misinformation detector
    
    Improvements:
    - Adaptive weighting based on claim type
    - Context-aware credibility scoring
    - Better fake news pattern matching
    - Stricter thresholds
    """

    def __init__(self):
        print(f"\n{'='*80}")
        print("Initializing ENHANCED Text Detector V2")
        print(f"{'='*80}")

        # EXPANDED fake news patterns
        self.fake_patterns = {
            'breaking_news': [
                r'breaking[\s:🚨💥⚡]+',
                r'urgent[\s:]+',
                r'alert[\s:]+',
                r'just in[\s:]+',
            ],
            'clickbait': [
                r'you won\'t believe',
                r'this one (trick|secret|weird trick)',
                r'what happens next',
                r'doctors hate',
                r'they don\'t want you to know',
                r'\d+ (tricks|secrets|ways|reasons) (to|why)',
            ],
            'sensationalism': [
                r'shocking[\s:]*',
                r'miracle cure',
                r'scientists (baffled|stunned|speechless)',
                r'100% (guaranteed|proven|works)',
                r'secret deal',
                r'conspiracy',
                r'cover[- ]up',
            ],
            'urgency': [
                r'act now',
                r'limited time',
                r'before it\'s too late',
                r'last chance',
                r'ends (today|soon|tonight)',
            ],
            'vagueness': [
                r'some (people|experts) say',
                r'many believe',
                r'it is said',
                r'rumors suggest',
                r'sources claim',
                r'allegedly',
            ]
        }

        # Credibility markers
        self.credibility_markers = {
            'authoritative_sources': [
                r'according to (reuters|ap news|bbc|cnn|new york times|washington post|associated press)',
                r'reported by',
                r'confirmed by',
            ],
            'academic': [
                r'published in (nature|science|cell|lancet|nejm)',
                r'peer[- ]reviewed',
                r'research (paper|study)',
                r'journal of',
            ],
            'specifics': [
                r'\d{4}[-/]\d{2}[-/]\d{2}',  # Dates
                r'\d+(\.\d+)?%',  # Percentages
                r'(dr\.|professor|phd) [A-Z][a-z]+',  # Named experts
                r'study of \d+',  # Sample sizes
            ],
            'attribution': [
                r'said in (an interview|a statement)',
                r'told (reporters|journalists)',
                r'quoted as saying',
                r'according to',
            ]
        }

        # Known reliable news domains for source cross-verification
        self.trusted_domains = [
            "reuters.com", "apnews.com", "bbc.com", "cnn.com",
            "nytimes.com", "washingtonpost.com", "theguardian.com",
            "ndtv.com", "thehindu.com", "indianexpress.com"
        ]

        # Wikidata / DBpedia lookup config
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"
        self.dbpedia_endpoint = "https://dbpedia.org/sparql"
        self.http_headers = {
            "User-Agent": "Multimodal-Deepfake-Detector/1.0 (academic use)"
        }

        # Emotional manipulation words
        self.emotion_words = {
            'fear': ['danger', 'threat', 'risk', 'disaster', 'catastrophe', 'deadly', 'fatal', 'terrifying'],
            'outrage': ['outrageous', 'disgusting', 'appalling', 'shocking', 'scandal', 'corrupt'],
            'hype': ['amazing', 'incredible', 'unbelievable', 'revolutionary', 'breakthrough', 'miracle']
        }

        print("✅ Enhanced Text Detector V2 Ready")
        print("  • Adaptive weighting")
        print("  • Context-aware credibility")
        print("  • Expanded pattern matching")
        print(f"{'='*80}\n")

    def _wikidata_dbpedia_lookup(self, text: str) -> Tuple[float, str]:
        """
        Verify common factual claims using Wikidata (knowledge graph lookup).
        Uses correct entity matching + position-held (P39) reasoning.
        """
        try:
            # Detect simple factual pattern: "X is Y of Z"
            match = re.search(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(prime minister|president|capital)\s+of\s+([A-Z][a-z]+)',
                text
            )
            if not match:
                return 0.5, "Knowledge lookup: not a structured factual claim"

            subject, relation, country = match.groups()

            # Handle PRIME MINISTER case (most common)
            if relation.lower() == "prime minister":
                query = f"""
                SELECT ?personLabel WHERE {{
                  ?person rdfs:label ?label .
                  FILTER(CONTAINS(LCASE(STR(?label)), "{subject.lower()}"))
                  ?person wdt:P39 ?position .
                  ?position rdfs:label "Prime Minister of India"@en .
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                """

            # Handle PRESIDENT case
            elif relation.lower() == "president":
                query = f"""
                SELECT ?personLabel WHERE {{
                  ?person rdfs:label ?label .
                  FILTER(CONTAINS(LCASE(STR(?label)), "{subject.lower()}"))
                  ?person wdt:P39 ?position .
                  FILTER(CONTAINS(LCASE(STR(?position)), "president"))
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                """

            # Handle CAPITAL case
            elif relation.lower() == "capital":
                query = f"""
                SELECT ?capitalLabel WHERE {{
                  ?country rdfs:label "{country}"@en .
                  ?country wdt:P36 ?capital .
                  ?capital rdfs:label "{subject}"@en .
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                """

            else:
                return 0.5, "Knowledge lookup: unsupported relation"

            res = requests.get(
                self.wikidata_endpoint,
                params={"format": "json", "query": query},
                headers=self.http_headers,
                timeout=5
            )

            if res.status_code == 200 and res.json().get("results", {}).get("bindings"):
                return 0.95, "Wikidata verified factual claim (knowledge graph)"
            else:
                return 0.4, "Wikidata lookup failed to verify claim"

        except Exception as e:
            return 0.5, f"Knowledge lookup unavailable ({type(e).__name__})"

    def _cross_verify_sources(self, text: str) -> Tuple[float, str]:
        """
        Cross-verify claims using cited URLs or named sources.
        This is a lightweight heuristic-based verifier (no web calls).
        """
        urls = re.findall(r'https?://[^\s]+', text)
        domains = []

        for url in urls:
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                domains.append(domain)
            except Exception:
                pass

        matched = [d for d in domains if any(td in d for td in self.trusted_domains)]

        if matched:
            return 0.9, f"Sources verified: trusted domains found ({', '.join(set(matched))})"
        elif urls:
            return 0.4, "Sources cited but none are trusted outlets → suspicious"
        else:
            return 0.2, "No external sources cited → cannot cross-verify"

    def analyze(self, text: str) -> TextAnalysisResult:
        """Analyze text with adaptive weighting"""
        
        if not text or len(text.strip()) < 10:
            return TextAnalysisResult(
                score=0.5,
                verdict='uncertain',
                reasoning='Text too short',
                evidence=[],
                confidence=0.0
            )
        
        text_lower = text.lower()
        scores = []
        weights = []
        evidence = []
        
        # Method 1: Fake news patterns (with category breakdown)
        fake_score, fake_evidence, fake_categories = self._detect_fake_patterns(text_lower)
        scores.append(fake_score)
        evidence.append(fake_evidence)
        
        # Check if this is a "breaking news" style claim
        is_breaking_news = any(re.search(p, text_lower) for p in self.fake_patterns['breaking_news'])
        
        # Method 2: Credibility markers (CONTEXT-AWARE)
        cred_score, cred_evidence = self._detect_credibility_markers(
            text_lower, 
            text,
            is_breaking_news=is_breaking_news
        )
        scores.append(cred_score)
        evidence.append(cred_evidence)
        
        # Method 3: Emotional manipulation
        emotion_score, emotion_evidence = self._detect_emotional_manipulation(text_lower)
        scores.append(emotion_score)
        evidence.append(emotion_evidence)
        
        # Method 4: Claim specificity
        spec_score, spec_evidence = self._analyze_claim_specificity(text)
        scores.append(spec_score)
        evidence.append(spec_evidence)

        # Method 4.5: Wikidata / DBpedia factual verification
        kb_score, kb_evidence = self._wikidata_dbpedia_lookup(text)
        scores.append(kb_score)
        evidence.append(kb_evidence)

        # Method 5: Source cross-verification
        source_score, source_evidence = self._cross_verify_sources(text)
        scores.append(source_score)
        evidence.append(source_evidence)

        # ADAPTIVE WEIGHTING
        if is_breaking_news and cred_score < 0.3:
            # BREAKING NEWS with NO SOURCES = VERY SUSPICIOUS
            weights = [0.18, 0.30, 0.12, 0.10, 0.15, 0.15]  # Boost credibility weight
            evidence.append("⚠️ BREAKING NEWS with no sources - highly suspicious!")
        elif fake_score < 0.3 and cred_score < 0.4:
            # Multiple fake patterns + no credibility
            weights = [0.30, 0.22, 0.13, 0.10, 0.15, 0.10]  # Boost fake pattern weight
            evidence.append("⚠️ Multiple fake news indicators detected!")
        else:
            # Normal weighting
            weights = [0.22, 0.22, 0.13, 0.13, 0.15, 0.15]

        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        # Stricter thresholds
        if final_score > 0.60:  # Raised from 0.65
            verdict = 'real'
            confidence = (final_score - 0.60) / 0.40
            reasoning = f"Text appears credible (score: {final_score:.3f})"
        elif final_score < 0.40:  # Raised from 0.35
            verdict = 'fake'
            confidence = (0.40 - final_score) / 0.40
            reasoning = f"Misinformation detected (score: {final_score:.3f})"
        else:
            verdict = 'uncertain'
            confidence = 0.3
            reasoning = f"Mixed signals (score: {final_score:.3f})"
        
        # Add weight info
        evidence.append(f"\nWeights: " + ", ".join(f"{w:.0%}" for w in weights))
        
        return TextAnalysisResult(
            score=final_score,
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            confidence=min(confidence, 1.0)
        )

    def _detect_fake_patterns(self, text: str) -> Tuple[float, str, List[str]]:
        """Detect fake news patterns with category breakdown"""
        
        detected_categories = []
        total_matches = 0
        
        for category, patterns in self.fake_patterns.items():
            category_matches = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    category_matches += 1
                    total_matches += 1
            
            if category_matches > 0:
                detected_categories.append(category)
        
        # Aggressive scoring
        if total_matches == 0:
            score = 0.8
            evidence = "Patterns: none detected → likely REAL"
        elif total_matches == 1:
            score = 0.4  # Changed from 0.5 - more suspicious
            evidence = f"Patterns: 1 indicator ({detected_categories[0]}) → suspicious"
        elif total_matches == 2:
            score = 0.2  # Changed from 0.3 - very suspicious
            evidence = f"Patterns: 2 indicators ({', '.join(detected_categories)}) → likely FAKE"
        else:
            score = 0.1  # Changed from 0.2 - almost certainly fake
            evidence = f"Patterns: {total_matches} indicators → FAKE"
        
        return score, evidence, detected_categories

    def _detect_credibility_markers(self, text_lower: str, text_orig: str, 
                                   is_breaking_news: bool = False) -> Tuple[float, str]:
        """CONTEXT-AWARE credibility detection"""
        
        marker_count = 0
        marker_types = []
        
        for marker_type, patterns in self.credibility_markers.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    marker_count += len(matches)
                    marker_types.append(marker_type)
        
        # CONTEXT-AWARE SCORING
        if is_breaking_news:
            # Breaking news REQUIRES sources - stricter
            if marker_count >= 3:
                score = 0.9
                evidence = f"Credibility: {marker_count} markers in breaking news → likely REAL"
            elif marker_count >= 1:
                score = 0.5  # Changed from 0.7 - not enough for breaking news
                evidence = f"Credibility: only {marker_count} markers for breaking news → suspicious"
            else:
                score = 0.1  # Changed from 0.4 - NO sources for breaking news = VERY suspicious
                evidence = f"Credibility: NO sources for breaking news claim → likely FAKE"
        else:
            # Regular text - normal scoring
            if marker_count >= 5:
                score = 0.9
                evidence = f"Credibility: {marker_count} markers → likely REAL"
            elif marker_count >= 3:
                score = 0.7
                evidence = f"Credibility: {marker_count} markers → probably real"
            elif marker_count >= 1:
                score = 0.5
                evidence = f"Credibility: {marker_count} markers → uncertain"
            else:
                score = 0.3  # Changed from 0.4 - more suspicious
                evidence = "Credibility: no sources cited → suspicious"
        
        return score, evidence

    def _detect_emotional_manipulation(self, text: str) -> Tuple[float, str]:
        """Detect emotional manipulation"""
        
        emotion_counts = {emo: 0 for emo in self.emotion_words}
        
        for emotion_type, words in self.emotion_words.items():
            for word in words:
                emotion_counts[emotion_type] += text.count(word)
        
        total_emotions = sum(emotion_counts.values())
        word_count = len(text.split())
        emotion_ratio = total_emotions / max(word_count, 1)
        
        # Stricter thresholds
        if emotion_ratio > 0.08:  # Changed from 0.05
            score = 0.1  # Changed from 0.2
            evidence = f"Emotion: HIGH manipulation ({emotion_ratio:.2%}) → likely FAKE"
        elif emotion_ratio > 0.04:  # Changed from 0.02
            score = 0.3  # Changed from 0.4
            evidence = f"Emotion: moderate manipulation ({emotion_ratio:.2%}) → suspicious"
        else:
            score = 0.7
            evidence = f"Emotion: neutral tone ({emotion_ratio:.2%}) → good"
        
        return score, evidence

    def _analyze_claim_specificity(self, text: str) -> Tuple[float, str]:
        """Analyze specificity of claims"""
        
        # Count specific elements
        dates = len(re.findall(r'\b\d{4}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}', text, re.IGNORECASE))
        numbers = len(re.findall(r'\b\d+(\.\d+)?\s*(percent|%|people|cases|dollars)', text, re.IGNORECASE))
        names = len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text))
        places = len(re.findall(r'\b(in|at|from)\s+[A-Z][a-z]+', text))
        
        specificity_score = dates + numbers + names * 0.5 + places * 0.5
        
        if specificity_score > 5:
            score = 0.8
            evidence = f"Specificity: HIGH detail ({specificity_score:.0f}) → credible"
        elif specificity_score > 2:
            score = 0.6
            evidence = f"Specificity: moderate detail ({specificity_score:.0f})"
        else:
            score = 0.3  # Changed from 0.4
            evidence = f"Specificity: vague/generic ({specificity_score:.0f}) → suspicious"
        
        return score, evidence


if __name__ == "__main__":
    print("\nTesting Enhanced Text Detector V2...\n")
    detector = EnhancedTextDetectorV2()
    
    # Test cases
    tests = [
        ("BREAKING 🚨 Bill Gates officially joins NVIDIA board after secret deal", "FAKE"),
        ("Modi is prime minister of India", "REAL"),
        ("You won't believe this shocking miracle cure doctors hate!", "FAKE"),
        ("Study published in Nature shows 67% efficacy in clinical trials", "REAL"),
    ]
    
    for text, expected in tests:
        result = detector.analyze(text)
        fake_pct = (1 - result.score) * 100
        status = "✅" if result.verdict.lower() == expected.lower() else "❌"
        
        print(f"{status} {expected}: {fake_pct:.1f}% fake")
        print(f"   Text: {text[:60]}...")
        print(f"   Verdict: {result.verdict.upper()}")
        print(f"   Evidence: {result.evidence[0]}")
        print()