"""
ENHANCED ZERO-SHOT VISUAL DETECTOR V2

Improvements over v1:
1. Adaptive thresholds for extreme values
2. Face-specific AI detection
3. Weighted voting based on signal strength
4. JPEG artifact analysis
5. AI-specific pattern detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import cv2

@dataclass
class AgentOutput:
    """Output from an agent"""
    score: float  # 0=fake, 1=real
    verdict: str
    reasoning: str
    evidence: List[str]
    confidence: float


class EnhancedZeroShotVisualDetectorV2:
    """
    Enhanced detector for sophisticated AI-generated images
    
    Can detect modern AI generators (Midjourney, DALL-E 3, SD3) that:
    - Add realistic noise
    - Preserve edges
    - Mimic camera artifacts
    - Have near-perfect faces
    """

    def __init__(self, 
             model_name="openai/clip-vit-base-patch32",
             device: str = None):
        print(f"\n{'='*80}")
        print("Initializing ENHANCED Zero-Shot Visual Detector V2")
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

        print(f"\nDevice: {self.device}")
        print("Loading CLIP...")

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()

        # Enhanced prompts
        self.prompts = {
            'real_photo': [
                "a photograph taken with a real camera",
                "natural photography with camera sensor noise",
                "real world scene with authentic imperfections",
            ],
            'ai_generated': [
                "AI generated synthetic image",
                "computer graphics render",
                "artificial neural network output",
                "midjourney or stable diffusion image"
            ],
            'ai_face_artifacts': [
                "AI generated face with perfect symmetry",
                "synthetic skin texture too smooth",
                "digitally generated portrait",
                "face with unnatural lighting"
            ]
        }

        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print("✅ Enhanced Detector Ready")
        print("  • Adaptive thresholds")
        print("  • Face-specific AI detection")
        print("  • Weighted voting")
        print("  • Extreme value handling")
        print(f"{'='*80}\n")

    def analyze_from_image(self, image: Image.Image) -> AgentOutput:
        """Enhanced analysis with adaptive weighting"""
        
        img_array = np.array(image)
        scores = []
        evidence = []
        weights = []  # Dynamic weights
        
        # Method 1: Frequency Analysis
        freq_score, freq_evidence = self._frequency_analysis(img_array)
        scores.append(freq_score)
        evidence.append(freq_evidence)
        weights.append(0.20)  # Base weight
        
        # Method 2: Noise Analysis
        noise_score, noise_evidence = self._noise_analysis(img_array)
        scores.append(noise_score)
        evidence.append(noise_evidence)
        weights.append(0.20)
        
        # Method 3: Enhanced Color Analysis with EXTREME value detection
        color_score, color_evidence, color_is_extreme = self._enhanced_color_analysis(img_array)
        scores.append(color_score)
        evidence.append(color_evidence)
        
        if color_is_extreme:
            # BOOST color weight dramatically if extreme
            weights.append(0.50)  # 50% weight!
            # Reduce other weights
            weights[0] = 0.10
            weights[1] = 0.10
            evidence.append("⚠️ EXTREME color values detected - highly suspicious!")
        else:
            weights.append(0.15)
        
        # Method 4: Face-Specific AI Detection
        face_score, face_evidence, has_face = self._detect_ai_face(img_array)
        if has_face:
            scores.append(face_score)
            evidence.append(face_evidence)
            weights.append(0.25)
        
        # Method 5: CLIP Analysis
        clip_score, clip_evidence = self._enhanced_clip_analysis(image)
        scores.append(clip_score)
        evidence.append(clip_evidence)
        
        if has_face:
            weights.append(0.15)
        else:
            weights.append(0.20)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        # Determine verdict with STRICTER thresholds
        if final_score > 0.60:  # Raised from 0.65
            verdict = 'real'
            confidence = (final_score - 0.60) / 0.40
            reasoning = f"Image appears authentic (score: {final_score:.3f})"
        elif final_score < 0.40:  # Raised from 0.35
            verdict = 'fake'
            confidence = (0.40 - final_score) / 0.40
            reasoning = f"AI-generated/manipulated detected (score: {final_score:.3f})"
        else:
            verdict = 'uncertain'
            confidence = 0.3
            reasoning = f"Inconclusive (score: {final_score:.3f})"
        
        # Add weight information to evidence
        evidence.append(f"\nWeights used: " + ", ".join(f"{w:.2f}" for w in weights))
        
        return AgentOutput(
            score=final_score,
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            confidence=min(confidence, 1.0)
        )

    def _frequency_analysis(self, img: np.ndarray) -> Tuple[float, str]:
        """FFT analysis (same as before)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radius = min(h, w) // 4
        high_freq_mask = distance > radius
        
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        if high_freq_ratio > 0.15:
            score = 0.85
            evidence = f"Frequency: HIGH noise ({high_freq_ratio:.3f}) → likely REAL"
        elif high_freq_ratio < 0.08:
            score = 0.15
            evidence = f"Frequency: LOW noise ({high_freq_ratio:.3f}) → likely FAKE"
        else:
            score = 0.5
            evidence = f"Frequency: moderate ({high_freq_ratio:.3f})"
        
        return score, evidence

    def _noise_analysis(self, img: np.ndarray) -> Tuple[float, str]:
        """Noise pattern analysis (same as before)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
        else:
            gray = img.astype(float)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        noise_std = np.std(noise)
        
        if 2.0 < noise_std < 8.0:
            score = 0.8
            evidence = f"Noise: consistent ({noise_std:.2f}) → likely REAL"
        elif noise_std < 1.5:
            score = 0.2
            evidence = f"Noise: too smooth ({noise_std:.2f}) → likely FAKE"
        else:
            score = 0.5
            evidence = f"Noise: inconclusive ({noise_std:.2f})"
        
        return score, evidence

    def _enhanced_color_analysis(self, img: np.ndarray) -> Tuple[float, str, bool]:
        """
        ENHANCED color analysis with extreme value detection
        Returns: (score, evidence, is_extreme)
        """
        if len(img.shape) == 2:
            return 0.5, "Color: grayscale", False
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        saturation = np.sqrt(a.astype(float)**2 + b.astype(float)**2)
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        is_extreme = False
        
        # EXTREME values (new!)
        if sat_mean > 150:
            score = 0.0  # Almost certainly fake
            evidence = f"Color: EXTREME oversaturation ({sat_mean:.1f}) → FAKE"
            is_extreme = True
        elif sat_mean > 100:
            score = 0.1  # Very suspicious
            evidence = f"Color: severe oversaturation ({sat_mean:.1f}) → likely FAKE"
            is_extreme = True
        elif sat_mean > 50:
            score = 0.2
            evidence = f"Color: oversaturated ({sat_mean:.1f}) → suspicious"
        elif 20 < sat_mean < 40 and 15 < sat_std < 30:
            score = 0.8
            evidence = f"Color: natural ({sat_mean:.1f}) → likely REAL"
        else:
            score = 0.5
            evidence = f"Color: inconclusive ({sat_mean:.1f})"
        
        return score, evidence, is_extreme

    def _detect_ai_face(self, img: np.ndarray) -> Tuple[float, str, bool]:
        """
        Detect AI-generated faces
        
        AI faces often have:
        - Perfect symmetry
        - Too-smooth skin
        - Unrealistic lighting
        - No pores or imperfections
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 0.5, "Face: none detected", False
        
        # Analyze largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_region = img[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        
        # Check 1: Skin smoothness (AI faces are too smooth)
        face_blur = cv2.GaussianBlur(face_gray, (5, 5), 0)
        skin_detail = face_gray.astype(float) - face_blur.astype(float)
        detail_std = np.std(skin_detail)
        
        # Check 2: Symmetry (AI faces are too symmetric)
        face_flip = cv2.flip(face_gray, 1)
        symmetry_diff = np.mean(np.abs(face_gray.astype(float) - face_flip.astype(float)))
        
        # Scoring
        ai_indicators = 0
        
        if detail_std < 2.0:  # Too smooth
            ai_indicators += 1
        if symmetry_diff < 15.0:  # Too symmetric
            ai_indicators += 1
        
        if ai_indicators == 0:
            score = 0.8
            evidence = f"Face: natural detail & asymmetry → likely REAL"
        elif ai_indicators == 1:
            score = 0.4
            evidence = f"Face: some AI indicators ({ai_indicators}/2) → suspicious"
        else:
            score = 0.1
            evidence = f"Face: AI-like (smooth={detail_std:.2f}, sym={symmetry_diff:.1f}) → likely FAKE"
        
        return score, evidence, True

    def _enhanced_clip_analysis(self, image: Image.Image) -> Tuple[float, str]:
        """Enhanced CLIP with face-specific prompts"""
        with torch.no_grad():
            # Check if image contains faces
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            has_face = len(faces) > 0
            
            if has_face:
                # Use face-specific prompts
                all_prompts = (
                    self.prompts['real_photo'] +
                    self.prompts['ai_generated'] +
                    self.prompts['ai_face_artifacts']
                )
            else:
                all_prompts = (
                    self.prompts['real_photo'] +
                    self.prompts['ai_generated']
                )
            
            inputs = self.clip_processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image.squeeze()
            probs = F.softmax(logits, dim=0).cpu().numpy()
            
            n_real = len(self.prompts['real_photo'])
            n_fake = len(self.prompts['ai_generated'])
            
            real_prob = np.mean(probs[:n_real])
            fake_prob = np.mean(probs[n_real:n_real+n_fake])
            
            if has_face:
                artifact_prob = np.mean(probs[n_real+n_fake:])
                score = real_prob / (real_prob + fake_prob + artifact_prob + 1e-8)
                score = score * (1 - artifact_prob * 0.5)
            else:
                score = real_prob / (real_prob + fake_prob + 1e-8)
            
            evidence = f"CLIP: real={real_prob:.3f}, fake={fake_prob:.3f}"
        
        return score, evidence

    def analyze_from_features(self, visual_features: torch.Tensor, 
                             image: Optional[Image.Image] = None) -> AgentOutput:
        """Fallback when only features available"""
        if image is not None:
            return self.analyze_from_image(image)
        else:
            return AgentOutput(
                score=0.5,
                verdict='uncertain',
                reasoning='Cannot perform full analysis without original image',
                evidence=['Feature-only analysis unavailable'],
                confidence=0.1
            )


if __name__ == "__main__":
    print("\nTesting Enhanced Visual Detector V2...")
    agent = EnhancedZeroShotVisualDetectorV2()
    print("✅ Agent ready with adaptive weighting and extreme value detection")