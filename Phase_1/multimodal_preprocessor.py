import torch
from typing import Dict, Optional, Union
from pathlib import Path
from config import Config
from text_preprocessor import TextPreprocessor
from Phase_1.image_preprocessor import ImagePreprocessor
from audio_preprocessor import AudioPreprocessor
from video_preprocessor import VideoPreprocessor

class MultimodalPreprocessor:
    """
    Unified interface for multimodal deepfake detection preprocessing
    Integrates text, image, audio, and video processing pipelines
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("Initializing Multimodal Deepfake Detection System - Phase 1")
        print("="*70 + "\n")
        
        Config.setup_directories()
        
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.audio_processor = AudioPreprocessor()
        self.video_processor = VideoPreprocessor()
        
        print("\n" + "="*70)
        print("✅ Phase 1 Complete: All Preprocessors Initialized Successfully!")
        print("="*70)
    
    def process_multimodal_input(self,
                                 text: Optional[str] = None,
                                 image_path: Optional[str] = None,
                                 audio_path: Optional[str] = None,
                                 video_path: Optional[str] = None) -> Dict:
        """
        Process multimodal input and extract features from all available modalities
        
        Args:
            text: Text content (news article, social media post, etc.)
            image_path: Path to image file
            audio_path: Path to audio file
            video_path: Path to video file
            
        Returns:
            Dictionary with features from all modalities
        """
        results = {
            'text_features': None,
            'image_features': None,
            'audio_features': None,
            'video_features': None,
            'modalities_present': []
        }
        
        # Process text
        if text:
            print("Processing text...")
            results['text_features'] = self.text_processor.extract_features(text)
            results['modalities_present'].append('text')
        
        # Process image
        if image_path:
            print("Processing image...")
            image = self.image_processor.load_image(image_path)
            results['image_features'] = self.image_processor.extract_hybrid_features(image)
            results['modalities_present'].append('image')
        
        # Process audio
        if audio_path:
            print("Processing audio...")
            results['audio_features'] = self.audio_processor.extract_hybrid_audio_features(audio_path)
            results['modalities_present'].append('audio')
        
        # Process video
        if video_path:
            print("Processing video...")
            results['video_features'] = self.video_processor.extract_spatiotemporal_features(video_path)
            results['modalities_present'].append('video')
        
        print(f"\n✅ Processed {len(results['modalities_present'])} modalities: {results['modalities_present']}")
        
        return results
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensionality of features from each modality"""
        return {
            'text': Config.TEXT_EMBEDDING_DIM,        # 768 (BERT)
            'image_vit': Config.IMAGE_EMBEDDING_DIM,   # 768 (ViT)
            'image_cnn': 1024,                         # ConvNeXt varies
            'audio_wav2vec': 768,                      # Wav2Vec2
            'video_swin': 1024                         # Swin Transformer
        }


if __name__ == "__main__":
    # Initialize the multimodal system
    preprocessor = MultimodalPreprocessor()
    
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY")
    print("="*70)
    print("\n📊 Feature Dimensions:")

    results = preprocessor.process_multimodal_input(
        text="This is a fake news article about politics.",
        image_path="Gemini_Generated_Image_1jzljk1jzljk1jzl.png",
        audio_path="file_example_WAV_1MG.wav",
        video_path="A.mp4"
    )

    print("\n🔍 Modalities processed:", results["modalities_present"])
    dims = preprocessor.get_feature_dimensions()
    for modality, dim in dims.items():
        print(f"   {modality:20s}: {dim} dimensions")
    
    print("\n🎯 Capabilities:")
    print("   ✅ Multilingual text analysis (5+ languages)")
    print("   ✅ Hybrid image analysis (ViT + ConvNeXt)")
    print("   ✅ Audio deepfake detection (Wav2Vec2 + MFCC)")
    print("   ✅ Video temporal consistency analysis")
    print("   ✅ Face detection and extraction")
    print("   ✅ Attention-based explainability (XAI)")
    print("   ✅ Batch processing support")
    
    print("\n🚀 Ready for Phase 2: Agentic Framework & Multimodal Fusion")
    print("="*70 + "\n")
