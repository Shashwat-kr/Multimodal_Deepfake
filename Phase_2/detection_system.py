import sys
from pathlib import Path
# Add project root (DeepFake/) to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import torch
import torch.nn as nn
from typing import Dict, Optional, List
import sys
from pathlib import Path

# Import Phase 1 preprocessors
from Phase_1.text_preprocessor import TextPreprocessor
from Phase_1.image_preprocessor import ImagePreprocessor
from Phase_1.audio_preprocessor import AudioPreprocessor
from Phase_1.video_preprocessor import VideoPreprocessor

# Import Phase 2 components
from Phase_2.multimodal_fusion import MultimodalTransformerFusion
from Phase_2.agentic_framework import (
    WebRetrievalAgent,
    AgentOutput
)
from Phase_2.zero_shot_visual_agent import ZeroShotVisualVeracityAgent
from Phase_2.zero_shot_consistency_agent import ZeroShotCrossModalConsistencyAgent
from Phase_2.zero_shot_reasoning_agent import ZeroShotReasoningAgent
from config import Config

class DeepfakeDetectionSystem(nn.Module):
    """
    Complete Multimodal Deepfake & Misinformation Detection System

    Integrates:
    - Phase 1: Transformer-based preprocessing (mBERT, ViT, ConvNeXt, Wav2Vec2, Swin)
    - Phase 2: Multimodal fusion with cross-attention
    - Phase 2: Agentic framework with 4 specialized agents
    - Explainable AI with Chain-of-Thought reasoning

    Target Performance: 96-97% accuracy based on literature survey
    """

    def __init__(self, 
                 fusion_dim: int = 512,
                 num_transformer_layers: int = 3):
        super().__init__()

        print("\n" + "="*80)
        print("Initializing Multimodal Deepfake Detection System")
        print("="*80 + "\n")

        # Phase 1: Preprocessors
        print("Loading Phase 1: Preprocessors...")
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.audio_processor = AudioPreprocessor()
        # Video processor loaded on-demand to save memory
        self._video_processor = None

        # Phase 2: Multimodal Fusion
        print("\nLoading Phase 2: Multimodal Fusion...")
        self.fusion_layer = MultimodalTransformerFusion(
            text_dim=768,
            image_dim=1792,
            audio_dim=768,
            video_dim=1024,
            fusion_dim=fusion_dim,
            num_heads=8,
            num_layers=num_transformer_layers
        )

        # Phase 2: Agentic Framework
        print("\nLoading Phase 2: Agentic Framework...")
        self.visual_agent = ZeroShotVisualVeracityAgent(
            model_name="openai/clip-vit-base-patch14",
            device=Config.DEVICE
        )
        self.consistency_agent = ZeroShotCrossModalConsistencyAgent(
            device=Config.DEVICE
        )
        self.web_agent = WebRetrievalAgent()
        self.reasoning_agent = ZeroShotReasoningAgent(
            visual_weight=0.35,      # 35% weight on visual
            consistency_weight=0.30,  # 30% on consistency
            web_weight=0.25,         # 25% on web verification
            fusion_weight=0.10,      # 10% on multimodal fusion
            device=Config.DEVICE
        )

        print("\n" + "="*80)
        print("✅ System Initialization Complete!")
        print("="*80 + "\n")

        self.device = Config.DEVICE
        self.to(self.device)

    @property
    def video_processor(self):
        """Lazy load video processor"""
        if self._video_processor is None:
            from Phase_1.video_preprocessor import VideoPreprocessor
            self._video_processor = VideoPreprocessor()
        return self._video_processor

    def detect(self,
               text: Optional[str] = None,
               image_path: Optional[str] = None,
               audio_path: Optional[str] = None,
               video_path: Optional[str] = None,
               return_detailed: bool = True) -> Dict:
        """
        Detect deepfakes and misinformation across modalities

        Args:
            text: Text content to analyze
            image_path: Path to image file
            audio_path: Path to audio file
            video_path: Path to video file
            return_detailed: Whether to return detailed explanation

        Returns:
            Detection results with verdict, confidence, and explanation
        """
        print("\n" + "="*80)
        print("DEEPFAKE DETECTION PIPELINE")
        print("="*80 + "\n")

        # Step 1: Feature Extraction
        print("Step 1: Extracting features from all modalities...")
        features = self._extract_features(text, image_path, audio_path, video_path)

        # Step 2: Multimodal Fusion
        print("\nStep 2: Fusing multimodal features...")
        fusion_output = self._fuse_features(features)

        # Step 3: Agentic Analysis
        print("\nStep 3: Running agentic analysis...")
        agent_outputs = self._run_agents(
            features, 
            text,
            image_path=image_path,  # ADDED
            video_path=video_path   # ADDED
        )

        # Step 4: Final Reasoning
        print("\nStep 4: Final reasoning and judgment...")
        final_output = self._final_reasoning(
            fusion_output['fused_features'],
            agent_outputs,
            text or ""
        )

        # Add modality information
        final_output['modalities_analyzed'] = features['modalities_present']
        final_output['modality_importance'] = fusion_output.get('modality_importance', {})

        print("\n" + "="*80)
        print(f"DETECTION COMPLETE: {final_output['verdict']}")
        print("="*80 + "\n")

        return final_output

    def _extract_features(self, text, image_path, audio_path, video_path) -> Dict:
        """Extract features from all available modalities"""
        features = {
            'text': None,
            'image': None,
            'audio': None,
            'video': None,
            'modalities_present': []
        }

        # Text features
        if text:
            print("   • Extracting text features (mBERT)...")
            features['text'] = self.text_processor.extract_features([text]).to(self.device)
            features['modalities_present'].append('text')

        # Image features
        if image_path:
            print("   • Extracting image features (ViT + ConvNeXt)...")
            image = self.image_processor.load_image(image_path)
            hybrid_features = self.image_processor.extract_hybrid_features(image)
            features['image'] = hybrid_features['fused_features'].to(self.device)
            features['modalities_present'].append('image')

        # Audio features
        if audio_path:
            print("   • Extracting audio features (Wav2Vec2)...")
            audio_features = self.audio_processor.extract_hybrid_audio_features(audio_path)
            features['audio'] = audio_features['wav2vec_features'].to(self.device)
            features['modalities_present'].append('audio')

        # Video features
        if video_path:
            print("   • Extracting video features (Swin Transformer)...")
            video_features = self.video_processor.extract_spatiotemporal_features(video_path)
            features['video'] = video_features['temporal_features'].to(self.device)
            features['modalities_present'].append('video')

        print(f"   ✅ Features extracted from {len(features['modalities_present'])} modalities")
        return features

    def _fuse_features(self, features: Dict) -> Dict:
        """Fuse multimodal features using transformer"""
        fusion_output = self.fusion_layer(
            text_features=features['text'],
            image_features=features['image'],
            audio_features=features['audio'],
            video_features=features['video'],
            return_attention=True
        )

        # Get modality importance
        if features['text'] is not None or features['image'] is not None:
            importance = self.fusion_layer.get_modality_importance(
                text_features=features['text'],
                image_features=features['image'],
                audio_features=features['audio'],
                video_features=features['video']
            )
            fusion_output['modality_importance'] = importance

            print("   ✅ Fusion complete")
            print("\n   Modality Importance Scores:")
            for modality, score in importance.items():
                print(f"      {modality.capitalize():10s}: {score:.4f}")

        return fusion_output

    def _run_agents(self, features: Dict, text: Optional[str],image_path: Optional[str] = None,
                video_path: Optional[str] = None) -> Dict:
        """Run all specialized agents"""
        agent_outputs = {}

        # Visual Veracity Agent - UPDATED
        print("  • Running Zero-Shot Visual Veracity Agent...")

        if image_path is not None:
            # Load PIL image for CLIP analysis
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            agent_outputs['visual'] = self.visual_agent.analyze_from_image(image)

        elif video_path is not None:
            # Extract a representative frame for video
            from PIL import Image
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                agent_outputs['visual'] = self.visual_agent.analyze_from_image(image)
            else:
                # Fallback to feature-based analysis
                visual_feat = features.get('video')
                if visual_feat is not None:
                    agent_outputs['visual'] = self.visual_agent.analyze_from_features(visual_feat)

        elif features.get('image') is not None:
            # Fallback: Use extracted features if image not available
            agent_outputs['visual'] = self.visual_agent.analyze_from_features(
                features['image']
            )

        else:
            # No visual content
            agent_outputs['visual'] = AgentOutput(
                score=0.5,
                verdict='uncertain',
                reasoning='No visual content available',
                evidence=[],
                confidence=0.0
            )

        print(f"   └─ Verdict: {agent_outputs['visual'].verdict.upper()} "
            f"(score: {agent_outputs['visual'].score:.3f})")

        # Cross-Modal Consistency Agent
        if len(features['modalities_present']) >= 2:
            print("  • Running Zero-Shot Cross-Modal Consistency Agent...")

            agent_outputs['consistency'] = self.consistency_agent.analyze(
                text_features=features['text'],
                visual_features=(features['image'] if features['image'] is not None 
                            else features['video']),
                audio_features=features['audio']
            )

            print(f"   └─ Verdict: {agent_outputs['consistency'].verdict.upper()} "
                f"(score: {agent_outputs['consistency'].score:.3f})")
        else:
            agent_outputs['consistency'] = AgentOutput(
                score=0.5, 
                verdict='uncertain', 
                reasoning='Single modality - cross-modal check not applicable',
                evidence=['Only one modality present'],
                confidence=0.0
            )
        # Web Retrieval Agent
        if text:
            print("   • Running Web Retrieval & Fact-Check Agent...")
            agent_outputs['web'] = self.web_agent.search_and_verify(text)
            print(f"     └─ Verdict: {agent_outputs['web'].verdict.upper()} (score: {agent_outputs['web'].score:.3f})")
        else:
            agent_outputs['web'] = AgentOutput(0.5, 'uncertain', 'No text to verify', [], 0.0)

        print("   ✅ All agents completed")
        return agent_outputs

    def _final_reasoning(self, 
                    fused_features: torch.Tensor,
                    agent_outputs: Dict, 
                    text: str) -> Dict:
        """
        Final reasoning with zero-shot weighted ensemble
        """
        # OLD: Used learned neural networks
        # NEW: Use weighted voting

        final_output = self.reasoning_agent.reason(
            fused_features=fused_features,
            visual_agent_output=agent_outputs['visual'],
            consistency_agent_output=agent_outputs['consistency'],
            web_agent_output=agent_outputs['web'],
            text_content=text
        )

        print(f"  🎯 Final Verdict: {final_output['verdict']}")
        print(f"  📊 Fake Probability: {final_output['fake_probability']:.3f}")
        print(f"  💪 Confidence: {final_output['confidence']:.3f}")
        print(f"  ⚠️  Risk Level: {final_output['risk_level']}")

        return final_output

    def batch_detect(self, samples: List[Dict]) -> List[Dict]:
        """
        Batch detection for multiple samples

        Args:
            samples: List of dictionaries with 'text', 'image_path', etc.

        Returns:
            List of detection results
        """
        results = []
        for i, sample in enumerate(samples):
            print(f"\nProcessing sample {i+1}/{len(samples)}...")
            result = self.detect(**sample)
            results.append(result)
        return results


if __name__ == "__main__":
    # Initialize system
    system = DeepfakeDetectionSystem()

    print("\n" + "="*80)
    print("SYSTEM READY FOR DETECTION")
    print("="*80)
    print("\nCapabilities:")
    print("  ✅ Text misinformation detection (mBERT multilingual)")
    print("  ✅ Image deepfake detection (ViT + ConvNeXt)")
    print("  ✅ Audio deepfake detection (Wav2Vec2 + MFCC)")
    print("  ✅ Video deepfake detection (Swin Transformer)")
    print("  ✅ Multimodal fusion (Transformer cross-attention)")
    print("  ✅ Agentic reasoning (4 specialized agents)")
    print("  ✅ Explainable AI (Chain-of-Thought)")

    print("\n" + "="*80)
    print("To use the system:")
    print("  result = system.detect(")
    print("      text='Your text here',")
    print("      image_path='path/to/image.jpg',")
    print("      audio_path='path/to/audio.wav'")
    print("  )")
    print("="*80 + "\n")
