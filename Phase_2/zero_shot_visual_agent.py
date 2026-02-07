import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

@dataclass
class AgentOutput:
    """Output from an agent"""
    score: float  # Confidence score (0-1)
    verdict: str  # 'real', 'fake', or 'uncertain'
    reasoning: str  # Explanation
    evidence: List[str]  # Supporting evidence
    confidence: float  # Confidence in the decision

class ZeroShotVisualVeracityAgent:
    """
    Zero-Shot Visual Veracity Agent using CLIP

    Uses pretrained CLIP model to detect deepfakes and AI-generated content
    without requiring any supervised training.

    Approach:
    1. Compare visual features against real/fake text prompts
    2. Analyze CLIP embedding distributions
    3. Check for AI generation artifacts using prompt engineering
    4. Multi-prompt ensemble for robust detection

    Reference: "Visual Language Models as Zero-Shot Deepfake Detectors"
    achieves superior performance with CLIP features.
    """

    def __init__(self, 
                 model_name: str = "openai/clip-vit-large-patch14",
                 device: str = None):
        """
        Initialize zero-shot visual agent

        Args:
            model_name: CLIP model variant (large recommended for accuracy)
            device: Device to run on (cuda/mps/cpu)
        """
        print(f"\n{'='*80}")
        print("Initializing Zero-Shot Visual Veracity Agent")
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
        print(f"🤖 Loading CLIP model: {model_name}...")

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze CLIP - no training needed!
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print("✅ CLIP model loaded and frozen (zero-shot mode)")

        # Define text prompts for zero-shot classification
        self.authenticity_prompts = {
            'real': [
                "a high quality authentic photograph",
                "a real unedited photo taken with a camera",
                "genuine photographic content",
                "natural lighting and realistic textures",
                "a photograph of real people or scenes"
            ],
            'fake': [
                "an AI-generated fake image",
                "computer generated imagery with artifacts",
                "synthetic deepfake content",
                "artificial image with inconsistent features",
                "manipulated photo with digital alterations"
            ]
        }

        # Artifact detection prompts
        self.artifact_prompts = [
            "blurry face boundaries",
            "unnatural skin texture",
            "inconsistent lighting",
            "distorted facial features",
            "digital compression artifacts",
            "misaligned eyes or teeth",
            "unrealistic hair rendering",
            "warped background",
            "color bleeding",
            "synthetic patterns"
        ]

        print(f"📋 Loaded {len(self.authenticity_prompts['real'])} real prompts")
        print(f"📋 Loaded {len(self.authenticity_prompts['fake'])} fake prompts")
        print(f"🔍 Loaded {len(self.artifact_prompts)} artifact detection prompts")
        print(f"{'='*80}\n")

    def analyze_from_features(self, 
                             visual_features: torch.Tensor,
                             image: Optional[Image.Image] = None) -> AgentOutput:
        """
        Analyze visual content from extracted features (for pipeline integration)

        Args:
            visual_features: Pre-extracted visual features (from ViT/ConvNeXt)
            image: Original PIL Image (optional, for CLIP analysis)

        Returns:
            AgentOutput with authenticity assessment
        """
        if image is not None:
            return self.analyze_from_image(image)
        else:
            # Fallback: Use heuristic analysis on raw features
            return self._analyze_features_heuristic(visual_features)

    def analyze_from_image(self, image: Image.Image) -> AgentOutput:
        """
        Analyze visual content directly from PIL Image using CLIP

        Args:
            image: PIL Image to analyze

        Returns:
            AgentOutput with detailed authenticity assessment
        """
        with torch.no_grad():
            # Step 1: Authenticity Classification
            authenticity_score = self._compute_authenticity_score(image)

            # Step 2: Artifact Detection
            artifact_score, detected_artifacts = self._detect_artifacts(image)

            # Step 3: Semantic Consistency Check
            consistency_score = self._check_semantic_consistency(image)

            # Step 4: Aggregate Scores
            # Weight: authenticity (50%), artifacts (30%), consistency (20%)
            final_score = (
                authenticity_score * 0.5 +
                (1 - artifact_score) * 0.3 +  # Lower artifacts = more real
                consistency_score * 0.2
            )

            # Determine verdict
            if final_score > 0.65 and artifact_score < 0.4:
                verdict = 'real'
                reasoning = (
                    f"Visual content appears authentic. "
                    f"CLIP authenticity score: {authenticity_score:.3f}, "
                    f"minimal artifacts detected ({artifact_score:.3f}), "
                    f"strong semantic consistency ({consistency_score:.3f})."
                )
            elif final_score < 0.35 or artifact_score > 0.6:
                verdict = 'fake'
                reasoning = (
                    f"Visual manipulation detected. "
                    f"CLIP authenticity score: {authenticity_score:.3f}, "
                    f"high artifact presence ({artifact_score:.3f}), "
                    f"inconsistent features ({consistency_score:.3f})."
                )
            else:
                verdict = 'uncertain'
                reasoning = (
                    f"Inconclusive visual analysis. "
                    f"CLIP authenticity score: {authenticity_score:.3f}, "
                    f"moderate artifact detection ({artifact_score:.3f})."
                )

            # Compile evidence
            evidence = [
                f"CLIP Authenticity Score: {authenticity_score:.3f}",
                f"Artifact Detection Score: {artifact_score:.3f}",
                f"Semantic Consistency: {consistency_score:.3f}",
                f"Top Artifacts: {', '.join(detected_artifacts[:3]) if detected_artifacts else 'None'}"
            ]

            # Confidence based on score extremity
            confidence = abs(final_score - 0.5) * 2

            return AgentOutput(
                score=final_score,
                verdict=verdict,
                reasoning=reasoning,
                evidence=evidence,
                confidence=confidence
            )

    def _compute_authenticity_score(self, image: Image.Image) -> float:
        """
        Compute authenticity score using CLIP zero-shot classification

        Returns:
            Score from 0 (fake) to 1 (real)
        """
        # Prepare all prompts
        all_prompts = (
            self.authenticity_prompts['real'] + 
            self.authenticity_prompts['fake']
        )

        # Process inputs
        inputs = self.clip_processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get CLIP similarities
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_prompts)
        probs = F.softmax(logits_per_image, dim=1).squeeze()

        # Split into real and fake probabilities
        num_real = len(self.authenticity_prompts['real'])
        real_probs = probs[:num_real]
        fake_probs = probs[num_real:]

        # Average probabilities for each class
        avg_real_prob = real_probs.mean().item()
        avg_fake_prob = fake_probs.mean().item()

        # Normalize to [0, 1] where 1 = real
        authenticity_score = avg_real_prob / (avg_real_prob + avg_fake_prob)

        return authenticity_score

    def _detect_artifacts(self, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Detect visual artifacts using CLIP prompt matching

        Returns:
            Tuple of (artifact_score, list of detected artifacts)
        """
        # Process with artifact prompts
        inputs = self.clip_processor(
            text=self.artifact_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image.squeeze()

        # Normalize scores
        artifact_scores = torch.sigmoid(logits / 100.0)  # Scale down logits

        # Identify detected artifacts (threshold at 0.5)
        detected = []
        for i, score in enumerate(artifact_scores):
            if score > 0.5:
                detected.append(self.artifact_prompts[i])

        # Overall artifact score (mean of all scores)
        overall_artifact_score = artifact_scores.mean().item()

        return overall_artifact_score, detected

    def _check_semantic_consistency(self, image: Image.Image) -> float:
        """
        Check semantic consistency of image features
        Uses diverse prompts to test if image has coherent semantics

        Returns:
            Consistency score from 0 to 1
        """
        consistency_prompts = [
            "coherent and logical scene composition",
            "consistent perspective and depth",
            "uniform lighting across the scene",
            "realistic object relationships"
        ]

        inputs = self.clip_processor(
            text=consistency_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image.squeeze()

        # Average normalized scores
        consistency_score = torch.sigmoid(logits / 100.0).mean().item()

        return consistency_score

    def _analyze_features_heuristic(self, 
                                   visual_features: torch.Tensor) -> AgentOutput:
        """
        Fallback heuristic analysis when only features available (no image)
        Uses statistical properties of feature embeddings

        Args:
            visual_features: (batch_size, feature_dim) tensor

        Returns:
            AgentOutput with heuristic assessment
        """
        with torch.no_grad():
            # Ensure 2D
            if visual_features.dim() == 1:
                visual_features = visual_features.unsqueeze(0)

            # Statistical analysis of features
            feature_std = visual_features.std(dim=1).mean().item()
            feature_mean_abs = visual_features.abs().mean().item()

            # Heuristic: Real images have moderate std and mean
            # Fake images often have unusual distributions
            std_score = 1.0 - abs(feature_std - 0.5) * 2  # Peak at 0.5 std
            mean_score = 1.0 - abs(feature_mean_abs - 0.3) * 3  # Peak near 0.3

            # Clamp to [0, 1]
            std_score = max(0.0, min(1.0, std_score))
            mean_score = max(0.0, min(1.0, mean_score))

            # Combined score
            heuristic_score = (std_score + mean_score) / 2

            if heuristic_score > 0.6:
                verdict = 'real'
                reasoning = "Feature distribution appears typical of authentic content"
            elif heuristic_score < 0.4:
                verdict = 'fake'
                reasoning = "Feature distribution shows anomalies consistent with manipulation"
            else:
                verdict = 'uncertain'
                reasoning = "Feature analysis inconclusive"

            evidence = [
                f"Feature std: {feature_std:.3f}",
                f"Feature mean abs: {feature_mean_abs:.3f}",
                f"Heuristic score: {heuristic_score:.3f}",
                "Note: Limited analysis without original image"
            ]

            return AgentOutput(
                score=heuristic_score,
                verdict=verdict,
                reasoning=reasoning,
                evidence=evidence,
                confidence=0.6  # Lower confidence for heuristic
            )

    def batch_analyze(self, images: List[Image.Image]) -> List[AgentOutput]:
        """
        Batch analysis for multiple images

        Args:
            images: List of PIL Images

        Returns:
            List of AgentOutput results
        """
        results = []
        for i, image in enumerate(images):
            print(f"Analyzing image {i+1}/{len(images)}...")
            result = self.analyze_from_image(image)
            results.append(result)
        return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Zero-Shot Visual Veracity Agent - Test Suite")
    print("="*80)

    # Initialize agent
    agent = ZeroShotVisualVeracityAgent()

    print("\n✅ Agent initialized successfully!")
    print("\nCapabilities:")
    print("  • Zero-shot deepfake detection using CLIP")
    print("  • Multi-prompt ensemble classification")
    print("  • Artifact detection and semantic consistency checks")
    print("  • Works with images OR pre-extracted features")
    print("  • No training data required!")
    print("\n" + "="*80 + "\n")
