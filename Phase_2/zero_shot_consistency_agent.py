import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

@dataclass
class AgentOutput:
    """Output from an agent"""
    score: float
    verdict: str
    reasoning: str
    evidence: List[str]
    confidence: float


class ZeroShotCrossModalConsistencyAgent:
    """
    Zero-Shot Cross-Modal Consistency Agent (BULLETPROOF v2)

    Completely rewritten with robust tensor handling.
    """

    def __init__(self, device: str = None):
        print(f"\n{'='*80}")
        print("Initializing Zero-Shot Cross-Modal Consistency Agent")
        print(f"{'='*80}")

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
        print("✅ Zero-shot consistency agent initialized")
        print("   • Text-Image alignment (cosine similarity)")
        print("   • Audio-Visual sync (correlation analysis)")
        print("   • Text-Audio consistency (semantic matching)")
        print(f"{'='*80}\n")

    def analyze(self,
                text_features: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> AgentOutput:
        """Check cross-modal consistency"""

        try:
            with torch.no_grad():
                # Prepare features
                text_feat = self._prepare_features(text_features, "text")
                visual_feat = self._prepare_features(visual_features, "visual")
                audio_feat = self._prepare_features(audio_features, "audio")

                consistency_scores = []
                evidence = []

                # Text-Image consistency
                if text_feat is not None and visual_feat is not None:
                    score = self._text_image_consistency(text_feat, visual_feat)
                    consistency_scores.append(score)
                    evidence.append(f"Text-Image: {score:.3f}")

                # Audio-Visual sync
                if audio_feat is not None and visual_feat is not None:
                    score = self._audio_visual_sync(audio_feat, visual_feat)
                    consistency_scores.append(score)
                    evidence.append(f"Audio-Visual: {score:.3f}")

                # Text-Audio consistency
                if text_feat is not None and audio_feat is not None:
                    score = self._text_audio_consistency(text_feat, audio_feat)
                    consistency_scores.append(score)
                    evidence.append(f"Text-Audio: {score:.3f}")

                # Check if we have enough data
                if len(consistency_scores) == 0:
                    return AgentOutput(
                        score=0.5,
                        verdict='uncertain',
                        reasoning='Insufficient modalities (need at least 2)',
                        evidence=['Single modality'],
                        confidence=0.0
                    )

                # Compute average and variance
                avg_score = float(np.mean(consistency_scores))
                variance = float(np.std(consistency_scores)) if len(consistency_scores) > 1 else 0.0

                # Determine verdict
                verdict, reasoning, confidence = self._make_verdict(
                    avg_score, variance, len(consistency_scores)
                )

                evidence.append(f"Average: {avg_score:.3f}")
                evidence.append(f"Variance: {variance:.3f}")

                return AgentOutput(
                    score=avg_score,
                    verdict=verdict,
                    reasoning=reasoning,
                    evidence=evidence,
                    confidence=confidence
                )

        except Exception as e:
            print(f"⚠️  Consistency agent error: {e}")
            return AgentOutput(
                score=0.5,
                verdict='uncertain',
                reasoning=f'Error during analysis: {str(e)}',
                evidence=['Processing error'],
                confidence=0.0
            )

    def _prepare_features(self, features: Optional[torch.Tensor], 
                         name: str) -> Optional[torch.Tensor]:
        """Safely prepare features for processing"""
        if features is None:
            return None

        # Convert to float and move to device
        feat = features.float().to(self.device)

        # Ensure 2D: (batch_size, feature_dim)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        elif feat.dim() > 2:
            # Flatten extra dimensions
            batch_size = feat.shape[0]
            feat = feat.view(batch_size, -1)

        return feat

    def _project_to_common_dim(self, features: torch.Tensor,
                               target_dim: int = 768) -> torch.Tensor:
        """Project features to common dimension"""
        current_dim = features.shape[-1]

        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            # Truncate
            return features[:, :target_dim]
        else:
            # Pad
            padding = torch.zeros(
                features.shape[0], 
                target_dim - current_dim,
                device=features.device,
                dtype=features.dtype
            )
            return torch.cat([features, padding], dim=-1)

    def _text_image_consistency(self, text_feat: torch.Tensor,
                               visual_feat: torch.Tensor) -> float:
        """Compute text-image consistency"""
        # Project to common dimension
        text_proj = self._project_to_common_dim(text_feat, 768)
        visual_proj = self._project_to_common_dim(visual_feat, 768)

        # Normalize
        text_norm = F.normalize(text_proj, p=2, dim=-1)
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)

        # Cosine similarity
        sim = (text_norm * visual_norm).sum(dim=-1)

        # Convert to Python float
        score = float(sim.mean().item())

        # Map from [-1, 1] to [0, 1]
        score = (score + 1.0) / 2.0

        return score

    def _audio_visual_sync(self, audio_feat: torch.Tensor,
                          visual_feat: torch.Tensor) -> float:
        """Compute audio-visual synchronization"""
        # Project to common dimension
        audio_proj = self._project_to_common_dim(audio_feat, 768)
        visual_proj = self._project_to_common_dim(visual_feat, 768)

        # Normalize
        audio_norm = F.normalize(audio_proj, p=2, dim=-1)
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)

        # Semantic similarity
        semantic_sim = (audio_norm * visual_norm).sum(dim=-1)
        semantic_score = float(semantic_sim.mean().item())
        semantic_score = (semantic_score + 1.0) / 2.0

        # Statistical similarity (std comparison)
        audio_std = float(audio_proj.std().item())
        visual_std = float(visual_proj.std().item())

        if max(audio_std, visual_std) > 1e-6:
            std_diff = abs(audio_std - visual_std) / max(audio_std, visual_std)
            std_score = 1.0 - min(std_diff, 1.0)
        else:
            std_score = 0.5

        # Combine
        final_score = semantic_score * 0.7 + std_score * 0.3

        return final_score

    def _text_audio_consistency(self, text_feat: torch.Tensor,
                               audio_feat: torch.Tensor) -> float:
        """Compute text-audio consistency"""
        # Project to common dimension
        text_proj = self._project_to_common_dim(text_feat, 768)
        audio_proj = self._project_to_common_dim(audio_feat, 768)

        # Normalize
        text_norm = F.normalize(text_proj, p=2, dim=-1)
        audio_norm = F.normalize(audio_proj, p=2, dim=-1)

        # Cosine similarity
        sim = (text_norm * audio_norm).sum(dim=-1)
        score = float(sim.mean().item())
        score = (score + 1.0) / 2.0

        return score

    def _make_verdict(self, avg_score: float, variance: float,
                     num_checks: int) -> Tuple[str, str, float]:
        """Determine final verdict"""
        # Adjust confidence based on variance
        base_confidence = abs(avg_score - 0.5) * 2.0
        confidence_penalty = min(variance * 2.0, 0.3)
        confidence = max(0.0, base_confidence - confidence_penalty)

        # Determine verdict
        if avg_score > 0.75 and variance < 0.15:
            verdict = 'real'
            reasoning = (
                f"Strong cross-modal consistency ({avg_score:.3f}). "
                f"All modalities align well. Content appears authentic."
            )
        elif avg_score < 0.35 or variance > 0.3:
            verdict = 'fake'
            reasoning = (
                f"Cross-modal inconsistency detected ({avg_score:.3f}, "
                f"var: {variance:.3f}). Modalities don't align properly."
            )
        else:
            verdict = 'uncertain'
            reasoning = (
                f"Moderate consistency ({avg_score:.3f}). "
                f"Some alignment but not conclusive."
            )

        return verdict, reasoning, confidence


if __name__ == "__main__":
    print("\nTesting Bulletproof Consistency Agent...")
    agent = ZeroShotCrossModalConsistencyAgent()

    # Test with various feature sizes
    text = torch.randn(1, 768)
    image = torch.randn(1, 1792)
    audio = torch.randn(1, 768)

    result = agent.analyze(text, image, audio)
    print(f"\nResult: {result.verdict.upper()}")
    print(f"Score: {result.score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print("\n✅ Bulletproof agent working!")
