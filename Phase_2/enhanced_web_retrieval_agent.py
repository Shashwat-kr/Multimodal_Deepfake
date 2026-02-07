import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import requests
from urllib.parse import quote_plus
import time

@dataclass
class AgentOutput:
    """Output from an agent"""
    score: float  # Confidence score (0-1)
    verdict: str  # 'real', 'fake', or 'uncertain'
    reasoning: str  # Explanation
    evidence: List[str]  # Supporting evidence
    confidence: float  # Confidence in the decision


class EnhancedWebRetrievalAgent:
    """
    Enhanced Web Retrieval & Fact-Check Agent

    Combines zero-shot pattern matching with real API calls:
    - Google Fact Check API (for claim verification)
    - SerpAPI / Google Search (for source credibility)
    - Heuristic-based fake news detection

    Falls back to pattern matching if APIs are unavailable.
    """

    def __init__(self, 
                 use_google_factcheck: bool = True,
                 use_web_search: bool = True,
                 google_api_key: Optional[str] = None,
                 serp_api_key: Optional[str] = None):
        """
        Initialize enhanced web retrieval agent

        Args:
            use_google_factcheck: Use Google Fact Check API
            use_web_search: Use web search for credibility
            google_api_key: Google API key (for Fact Check API)
            serp_api_key: SerpAPI key (for Google Search)
        """
        print(f"\n{'='*80}")
        print("Initializing Enhanced Web Retrieval & Fact-Check Agent")
        print(f"{'='*80}")

        self.use_google_factcheck = use_google_factcheck
        self.use_web_search = use_web_search
        self.google_api_key = google_api_key
        self.serp_api_key = serp_api_key

        # Fake news indicators (zero-shot pattern matching)
        self.fake_indicators = [
            r'\bbreaking\b.*\bshocking\b',
            r'\byou won't believe\b',
            r'\bdoctors hate (him|her|this)\b',
            r'\bmiracle cure\b',
            r'\b100% (guaranteed|proven|effective)\b',
            r'\bthis one (trick|tip|secret)\b',
            r'\bclick here to\b',
            r'\bact now\b.*\blimited time\b',
            r'\bthey don't want you to know\b',
            r'\bscientists shocked\b'
        ]

        # Credible source patterns
        self.credible_sources = [
            r'\b(reuters|ap news|bbc|cnn|nytimes|washingtonpost)\b',
            r'\b(nature|science|cell|lancet)\.com\b',
            r'\b(gov|edu)\.\w+\b',
            r'\bpeer[- ]reviewed\b',
            r'\bpublished in\b'
        ]

        print(f"\n🔍 Configuration:")
        print(f"   Google Fact Check API: {'✅ Enabled' if use_google_factcheck and google_api_key else '❌ Disabled'}")
        print(f"   Web Search: {'✅ Enabled' if use_web_search else '❌ Disabled'}")
        print(f"   Pattern Matching: ✅ Enabled (fallback)")

        print(f"\n✅ Enhanced web retrieval agent initialized")
        print(f"{'='*80}\n")

    def search_and_verify(self, text: str) -> AgentOutput:
        """
        Search web and verify claim credibility

        Args:
            text: Text content to verify

        Returns:
            AgentOutput with verification result
        """
        if not text or len(text.strip()) < 10:
            return AgentOutput(
                score=0.5, verdict='uncertain',
                reasoning='Text too short for verification',
                evidence=[], confidence=0.0
            )

        evidence = []
        scores = []

        # Method 1: Google Fact Check API
        if self.use_google_factcheck and self.google_api_key:
            factcheck_result = self._google_factcheck(text)
            if factcheck_result:
                scores.append(factcheck_result['score'])
                evidence.append(factcheck_result['evidence'])

        # Method 2: Web Search Credibility
        if self.use_web_search:
            if self.serp_api_key:
                search_result = self._web_search_credibility(text)
                if search_result:
                    scores.append(search_result['score'])
                    evidence.append(search_result['evidence'])
            else:
                # Fallback: Simple credible source check
                source_score = self._check_credible_sources(text)
                scores.append(source_score)
                evidence.append(f"Source credibility: {source_score:.2f}")

        # Method 3: Pattern-based fake news detection (always runs)
        pattern_score = self._pattern_based_detection(text)
        scores.append(pattern_score)
        evidence.append(f"Pattern analysis: {pattern_score:.2f}")

        # Aggregate scores
        if scores:
            avg_score = sum(scores) / len(scores)
            confidence = min(len(scores) / 3.0, 1.0)  # More methods = higher confidence
        else:
            avg_score = 0.5
            confidence = 0.0

        # Determine verdict
        if avg_score > 0.7:
            verdict = 'real'
            reasoning = f"Content appears credible. Average verification score: {avg_score:.3f}"
        elif avg_score < 0.4:
            verdict = 'fake'
            reasoning = f"Content shows signs of misinformation. Average verification score: {avg_score:.3f}"
        else:
            verdict = 'uncertain'
            reasoning = f"Mixed signals detected. Average verification score: {avg_score:.3f}"

        return AgentOutput(
            score=avg_score,
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            confidence=confidence
        )

    def _google_factcheck(self, text: str) -> Optional[Dict]:
        """
        Use Google Fact Check API to verify claims
        """
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'key': self.google_api_key,
                'query': text[:500],  # Limit query length
                'languageCode': 'en'
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if 'claims' in data and data['claims']:
                    # Analyze fact check ratings
                    ratings = []
                    for claim in data['claims'][:3]:  # Top 3 results
                        for review in claim.get('claimReview', []):
                            rating = review.get('textualRating', '').lower()

                            # Map ratings to scores
                            if any(word in rating for word in ['false', 'fake', 'pants on fire']):
                                ratings.append(0.1)
                            elif any(word in rating for word in ['mostly false', 'misleading']):
                                ratings.append(0.3)
                            elif any(word in rating for word in ['mixture', 'half true']):
                                ratings.append(0.5)
                            elif any(word in rating for word in ['mostly true', 'true']):
                                ratings.append(0.8)

                    if ratings:
                        score = sum(ratings) / len(ratings)
                        return {
                            'score': score,
                            'evidence': f"Google Fact Check: {len(ratings)} claims reviewed, avg rating: {score:.2f}"
                        }

            return None

        except Exception as e:
            print(f"   ⚠️  Google Fact Check API error: {e}")
            return None

    def _web_search_credibility(self, text: str) -> Optional[Dict]:
        """
        Use SerpAPI to search and check source credibility
        """
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': self.serp_api_key,
                'q': text[:200],
                'num': 5
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if 'organic_results' in data:
                    credible_count = 0
                    total_count = len(data['organic_results'])

                    for result in data['organic_results']:
                        link = result.get('link', '').lower()

                        # Check if source is credible
                        for pattern in self.credible_sources:
                            if re.search(pattern, link, re.IGNORECASE):
                                credible_count += 1
                                break

                    score = credible_count / total_count if total_count > 0 else 0.5

                    return {
                        'score': score,
                        'evidence': f"Web search: {credible_count}/{total_count} credible sources"
                    }

            return None

        except Exception as e:
            print(f"   ⚠️  SerpAPI error: {e}")
            return None

    def _check_credible_sources(self, text: str) -> float:
        """
        Simple check for credible source mentions in text
        """
        text_lower = text.lower()

        credible_count = 0
        for pattern in self.credible_sources:
            if re.search(pattern, text_lower):
                credible_count += 1

        # Score based on credible source mentions
        score = min(credible_count / 3.0, 1.0)
        return 0.5 + score * 0.3  # Range: 0.5 to 0.8

    def _pattern_based_detection(self, text: str) -> float:
        """
        Zero-shot pattern-based fake news detection
        """
        text_lower = text.lower()

        # Count fake indicators
        fake_count = 0
        for pattern in self.fake_indicators:
            if re.search(pattern, text_lower):
                fake_count += 1

        # Score: more fake indicators = lower credibility
        if fake_count == 0:
            score = 0.7  # No red flags
        elif fake_count == 1:
            score = 0.5  # One red flag
        elif fake_count == 2:
            score = 0.3  # Two red flags
        else:
            score = 0.1  # Multiple red flags = likely fake

        return score


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Enhanced Web Retrieval Agent")
    print("="*80)

    # Test without API keys (pattern matching only)
    agent = EnhancedWebRetrievalAgent(
        use_google_factcheck=False,
        use_web_search=False
    )

    # Test 1: Fake news indicators
    print("\n" + "="*80)
    print("TEST 1: Text with Fake News Indicators")
    print("="*80)

    fake_text = "BREAKING: Shocking miracle cure discovered! Doctors hate this one trick! 100% guaranteed!"
    result = agent.search_and_verify(fake_text)

    print(f"\nText: {fake_text}")
    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Reasoning: {result.reasoning}")

    # Test 2: Credible source mention
    print("\n" + "="*80)
    print("TEST 2: Text with Credible Sources")
    print("="*80)

    real_text = "According to Reuters and published in Nature, the peer-reviewed study shows..."
    result = agent.search_and_verify(real_text)

    print(f"\nText: {real_text}")
    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Reasoning: {result.reasoning}")

    print("\n" + "="*80)
    print("✅ Enhanced Web Retrieval Agent: TESTS COMPLETE")
    print("="*80 + "\n")
