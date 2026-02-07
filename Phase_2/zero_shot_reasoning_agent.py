import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AgentOutput:
    """Output from an agent"""
    score: float  # Confidence score (0-1)
    verdict: str  # 'real', 'fake', or 'uncertain'
    reasoning: str  # Explanation
    evidence: List[str]  # Supporting evidence
    confidence: float  # Confidence in the decision


class ZeroShotReasoningAgent:
    """
    Zero-Shot Final Reasoning & Judgment Agent (TUNED)

    Enhanced confidence calculation for better real-world performance.
    """

    def __init__(self,
                 visual_weight: float = 0.35,
                 consistency_weight: float = 0.30,
                 web_weight: float = 0.25,
                 fusion_weight: float = 0.10,
                 device: str = None):
        """Initialize zero-shot reasoning agent with tuned confidence"""
        print(f"\n{'='*80}")
        print("Initializing Zero-Shot Reasoning Agent (TUNED)")
        print(f"{'='*80}")

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"\n📱 Device: {self.device}")

        # Agent weights
        self.weights = {
            'visual': visual_weight,
            'consistency': consistency_weight,
            'web': web_weight,
            'fusion': fusion_weight
        }

        # Validate and normalize weights
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            print(f"⚠️  Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total_weight

        print("\n⚖️  Agent Weights:")
        for agent, weight in self.weights.items():
            print(f"   {agent.capitalize():15s}: {weight:.2%}")

        # TUNED: Adjusted thresholds for better performance
        self.fake_threshold = 0.60  # Lowered from 0.65 (easier to detect fake)
        self.real_threshold = 0.40  # Raised from 0.35 (easier to detect real)

        # TUNED: Lower confidence threshold for verdicts
        self.high_confidence_threshold = 0.70  # Down from 0.75
        self.low_confidence_threshold = 0.35   # Down from 0.40

        print(f"\n🎯 Verdict Thresholds (TUNED):")
        print(f"   FAKE if score > {self.fake_threshold:.2f}")
        print(f"   REAL if score < {self.real_threshold:.2f}")
        print(f"   UNCERTAIN otherwise")
        print(f"   Min confidence for verdict: {self.low_confidence_threshold:.2f}")

        print("\n✅ Zero-shot reasoning agent initialized (tuned for better confidence)")
        print(f"{'='*80}\n")

    def reason(self,
               fused_features: Optional[torch.Tensor],
               visual_agent_output: AgentOutput,
               consistency_agent_output: AgentOutput,
               web_agent_output: AgentOutput,
               text_content: str = "") -> Dict:
        """Perform final reasoning with tuned confidence calculation"""

        # Collect agent scores
        agent_scores = {
            'visual': visual_agent_output.score,
            'consistency': consistency_agent_output.score,
            'web': web_agent_output.score
        }

        agent_confidences = {
            'visual': visual_agent_output.confidence,
            'consistency': consistency_agent_output.confidence,
            'web': web_agent_output.confidence
        }

        agent_verdicts = {
            'visual': visual_agent_output.verdict,
            'consistency': consistency_agent_output.verdict,
            'web': web_agent_output.verdict
        }

        # Compute fusion score
        if fused_features is not None:
            fusion_score = self._compute_fusion_score(fused_features)
        else:
            fusion_score = 0.5

        agent_scores['fusion'] = fusion_score

        # Weighted ensemble score
        weighted_score = self._compute_weighted_score(agent_scores)

        # Agreement analysis
        agreement_score, agreement_analysis = self._analyze_agreement(
            agent_verdicts, agent_confidences
        )

        # TUNED: Final verdict & confidence
        final_verdict, confidence = self._determine_verdict_tuned(
            weighted_score, agreement_score, agent_confidences
        )

        # Risk level assessment
        risk_level = self._assess_risk_level(weighted_score, confidence)

        # Generate explanation
        explanation = self._generate_explanation(
            visual_agent_output,
            consistency_agent_output,
            web_agent_output,
            agent_scores,
            weighted_score,
            confidence,
            agreement_analysis
        )

        return {
            'verdict': final_verdict,
            'fake_probability': weighted_score,
            'confidence': confidence,
            'risk_level': risk_level,
            'explanation': explanation,
            'agent_scores': agent_scores,
            'agent_verdicts': agent_verdicts,
            'agreement_score': agreement_score,
            'agreement_analysis': agreement_analysis,
            'agent_outputs': {
                'visual_veracity': visual_agent_output,
                'cross_modal_consistency': consistency_agent_output,
                'web_verification': web_agent_output
            }
        }

    def _compute_weighted_score(self, agent_scores: Dict[str, float]) -> float:
        """Compute weighted ensemble score"""
        weighted_sum = 0.0

        for agent, score in agent_scores.items():
            weight = self.weights.get(agent, 0.0)

            # Convert to fake probability (flip for visual, consistency, web)
            if agent in ['visual', 'consistency', 'web']:
                fake_prob = 1.0 - score
            else:
                fake_prob = score

            weighted_sum += weight * fake_prob

        return weighted_sum

    def _compute_fusion_score(self, fused_features: torch.Tensor) -> float:
        """Compute fusion score from multimodal features"""
        with torch.no_grad():
            if fused_features.dim() == 1:
                fused_features = fused_features.unsqueeze(0)

            fused_features = fused_features.float()

            feature_mean = fused_features.mean().item()
            feature_std = fused_features.std().item()
            feature_max = fused_features.max().item()

            mean_score = 1.0 - abs(feature_mean) * 2
            std_score = min(feature_std / 0.5, 1.0)
            max_score = 1.0 if feature_max < 3.0 else 0.5

            fusion_score = (mean_score + std_score + max_score) / 3
            fusion_score = max(0.0, min(1.0, fusion_score))

            return fusion_score

    def _analyze_agreement(self,
                          agent_verdicts: Dict[str, str],
                          agent_confidences: Dict[str, float]) -> tuple:
        """Analyze agreement between agents"""
        verdicts_list = [v for v in agent_verdicts.values() if v != 'uncertain']

        if not verdicts_list:
            return 0.5, {'agreement': 'low', 'reason': 'All agents uncertain'}

        fake_votes = verdicts_list.count('fake')
        real_votes = verdicts_list.count('real')
        total_votes = len(verdicts_list)

        # Agreement score
        if fake_votes == total_votes or real_votes == total_votes:
            agreement_score = 1.0
            agreement_type = 'unanimous'
        elif max(fake_votes, real_votes) >= total_votes * 0.66:
            agreement_score = 0.75
            agreement_type = 'strong_majority'
        elif max(fake_votes, real_votes) > total_votes * 0.5:
            agreement_score = 0.5
            agreement_type = 'majority'
        else:
            agreement_score = 0.25
            agreement_type = 'split'

        # TUNED: Boost agreement score more with confidence
        avg_confidence = np.mean(list(agent_confidences.values()))
        agreement_score *= (0.4 + avg_confidence * 0.6)  # Changed from 0.5 + 0.5

        analysis = {
            'agreement': agreement_type,
            'fake_votes': fake_votes,
            'real_votes': real_votes,
            'total_votes': total_votes,
            'avg_confidence': avg_confidence
        }

        return agreement_score, analysis

    def _determine_verdict_tuned(self,
                                 weighted_score: float,
                                 agreement_score: float,
                                 agent_confidences: Dict[str, float]) -> tuple:
        """TUNED: Determine final verdict with improved confidence"""
        avg_confidence = np.mean(list(agent_confidences.values()))

        # TUNED: More aggressive confidence calculation
        base_confidence = abs(weighted_score - 0.5) * 2

        # TUNED: Give more weight to agreement and individual confidence
        confidence = base_confidence * (0.5 + agreement_score * 0.3 + avg_confidence * 0.2)

        # TUNED: Boost confidence if score is very extreme
        if weighted_score > 0.75 or weighted_score < 0.25:
            confidence = min(confidence * 1.15, 1.0)  # 15% boost

        # Clamp
        confidence = max(0.0, min(1.0, confidence))

        # TUNED: Determine verdict with adjusted thresholds
        if weighted_score > self.fake_threshold and confidence > self.low_confidence_threshold:
            verdict = 'FAKE'
        elif weighted_score < self.real_threshold and confidence > self.low_confidence_threshold:
            verdict = 'REAL'
        else:
            verdict = 'UNCERTAIN'

        return verdict, confidence

    def _assess_risk_level(self, score: float, confidence: float) -> str:
        """Assess risk level of content"""
        if score > 0.75 and confidence > 0.6:
            return 'HIGH'
        elif score > 0.55 and confidence > 0.45:
            return 'MEDIUM'
        elif score < 0.35:
            return 'LOW'
        else:
            return 'MEDIUM'

    def _generate_explanation(self,
                             visual_output: AgentOutput,
                             consistency_output: AgentOutput,
                             web_output: AgentOutput,
                             agent_scores: Dict[str, float],
                             final_score: float,
                             confidence: float,
                             agreement_analysis: Dict) -> str:
        """Generate Chain-of-Thought explanation"""
        explanation = "\n=== STEP-BY-STEP REASONING ===\n\n"

        # Step 1: Visual Analysis
        explanation += "1. VISUAL ANALYSIS:\n"
        explanation += f"   Verdict: {visual_output.verdict.upper()}\n"
        explanation += f"   {visual_output.reasoning}\n"
        if visual_output.evidence:
            explanation += f"   Evidence: {visual_output.evidence[0]}\n\n"
        else:
            explanation += "\n"

        # Step 2: Cross-Modal Consistency
        explanation += "2. CROSS-MODAL CONSISTENCY:\n"
        explanation += f"   Verdict: {consistency_output.verdict.upper()}\n"
        explanation += f"   {consistency_output.reasoning}\n"
        if consistency_output.evidence:
            explanation += f"   Evidence: {consistency_output.evidence[0]}\n\n"
        else:
            explanation += "\n"

        # Step 3: Web Verification
        explanation += "3. WEB VERIFICATION:\n"
        explanation += f"   Verdict: {web_output.verdict.upper()}\n"
        explanation += f"   {web_output.reasoning}\n"
        if web_output.evidence:
            explanation += f"   Evidence: {web_output.evidence[0]}\n\n"
        else:
            explanation += "\n"

        # Step 4: Agreement Analysis
        explanation += "4. AGENT AGREEMENT:\n"
        agreement_type = agreement_analysis['agreement']
        fake_votes = agreement_analysis['fake_votes']
        real_votes = agreement_analysis['real_votes']
        explanation += f"   Agreement: {agreement_type.replace('_', ' ').title()}\n"
        explanation += f"   Votes: {fake_votes} FAKE, {real_votes} REAL\n\n"

        # Step 5: Final Judgment
        explanation += "5. FINAL JUDGMENT:\n"
        explanation += f"   After aggregating all evidence with weighted ensemble voting,\n"
        explanation += f"   the content is classified as "

        if final_score > 0.5:
            explanation += f"FAKE with {confidence*100:.1f}% confidence.\n"
        else:
            explanation += f"REAL with {confidence*100:.1f}% confidence.\n"

        explanation += f"   Fake probability: {final_score:.3f}\n"
        explanation += f"   Agent scores: Visual={agent_scores['visual']:.3f}, "
        explanation += f"Consistency={agent_scores['consistency']:.3f}, "
        explanation += f"Web={agent_scores['web']:.3f}\n"

        return explanation


if __name__ == "__main__":
    print("Testing TUNED Zero-Shot Reasoning Agent...")
    agent = ZeroShotReasoningAgent()
    print("\n✅ Tuned agent ready with improved confidence calculation")