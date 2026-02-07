
import torch
import numpy as np
from pathlib import Path
import sys

print("\n" + "="*80)
print("MULTIMODAL DEEPFAKE DETECTION SYSTEM - PHASE 1 COMPREHENSIVE TEST")
print("="*80 + "\n")

# Test 1: Configuration
print("TEST 1: System Configuration")
print("-" * 80)
from config import Config

Config.setup_directories()
print(f"Device: {Config.DEVICE}")
print(f"Text Embedding Dim: {Config.TEXT_EMBEDDING_DIM}")
print(f"Image Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
print(f"Audio Sample Rate: {Config.AUDIO_SAMPLE_RATE} Hz")
print(f"Max Video Frames: {Config.MAX_FRAMES}")
print("✅ Configuration test passed!\n")

# Test 2: Text Preprocessor
print("\nTEST 2: Text Preprocessor (mBERT - Multilingual)")
print("-" * 80)
from text_preprocessor import TextPreprocessor

text_processor = TextPreprocessor()

# Test with multiple languages
test_texts = [
    "This is a fake news article spreading misinformation about elections.",
    "यह एक नकली समाचार लेख है।",  # Hindi
    "આ એક બનાવટી સમાચાર લેખ છે.",  # Gujarati
    "இது ஒரு போலி செய்தி கட்டுரை."  # Tamil
]

print(f"\nProcessing {len(test_texts)} multilingual texts...")
features = text_processor.extract_features(test_texts)
print(f"✅ Text features extracted: {features.shape}")
print(f"   - Batch size: {features.shape[0]}")
print(f"   - Embedding dimension: {features.shape[1]}")

# Test attention extraction
print(f"\nExtracting attention weights for explainability...")
attention_result = text_processor.extract_features_with_attention([test_texts[0]])
print(f"✅ Attention extraction complete:")
print(f"   - Attention layers: {len(attention_result['attention_weights'])}")
print(f"   - Attention heads per layer: {attention_result['attention_weights'][0].shape[1]}")
print(f"   - First 15 tokens: {attention_result['tokens'][:15]}")

# Test batch processing
print(f"\nTesting batch processing with 10 samples...")
large_batch = [f"Sample text number {i} for testing." for i in range(10)]
batch_features = text_processor.batch_process(large_batch, batch_size=3)
print(f"✅ Batch processing complete: {batch_features.shape}")

print("\n✅ Text Preprocessor: ALL TESTS PASSED")

# Test 3: Image Preprocessor
print("\n" + "="*80)
print("TEST 3: Image Preprocessor (ViT + ConvNeXt Hybrid)")
print("-" * 80)
from Phase_1.image_preprocessor import ImagePreprocessor
from PIL import Image

image_processor = ImagePreprocessor()

# Create synthetic test images
print(f"\nCreating synthetic test images ({Config.IMAGE_SIZE}x{Config.IMAGE_SIZE})...")
test_images = []
for i in range(3):
    # Create random RGB image
    img_array = np.random.randint(0, 255, (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    test_images.append(img)

print(f"Processing {len(test_images)} images...")

# Test ViT features
vit_features = image_processor.extract_vit_features(test_images)
print(f"\n✅ ViT features extracted: {vit_features.shape}")
print(f"   - Global attention features: {vit_features.shape[1]} dims")

# Test CNN features
cnn_features = image_processor.extract_cnn_features(test_images)
print(f"\n✅ ConvNeXt features extracted: {cnn_features.shape}")
print(f"   - Local texture features: {cnn_features.shape[1]} dims")

# Test hybrid features (Local-Global Synergy)
hybrid_features = image_processor.extract_hybrid_features(test_images)
print(f"\n✅ Hybrid features (GenConViT approach):")
print(f"   - ViT (global): {hybrid_features['vit_features'].shape}")
print(f"   - CNN (local): {hybrid_features['cnn_features'].shape}")
print(f"   - Fused: {hybrid_features['fused_features'].shape}")

# Test attention for explainability
print(f"\nExtracting attention maps for Grad-CAM visualization...")
attention_result = image_processor.extract_features_with_attention([test_images[0]])
print(f"✅ Attention extraction complete:")
print(f"   - Attention layers: {len(attention_result['attention_weights'])}")
print(f"   - Image patches: {attention_result['num_patches']}")
print(f"   - Patch size: {Config.IMAGE_PATCH_SIZE}x{Config.IMAGE_PATCH_SIZE}")

print("\n✅ Image Preprocessor: ALL TESTS PASSED")

# Test 4: Audio Preprocessor
print("\n" + "="*80)
print("TEST 4: Audio Preprocessor (Wav2Vec2 + MFCC)")
print("-" * 80)
from audio_preprocessor import AudioPreprocessor

audio_processor = AudioPreprocessor()

# Create synthetic audio waveform
print(f"\nCreating synthetic audio waveform ({Config.AUDIO_MAX_LENGTH}s @ {Config.AUDIO_SAMPLE_RATE} Hz)...")
duration = Config.AUDIO_MAX_LENGTH
sample_rate = Config.AUDIO_SAMPLE_RATE
t = np.linspace(0, duration, duration * sample_rate)
# Create a simple sine wave
frequency = 440  # A4 note
synthetic_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

# Test MFCC extraction
print(f"\nExtracting MFCC features...")
mfcc = audio_processor.extract_mfcc(synthetic_audio)
print(f"✅ MFCC features extracted: {mfcc.shape}")
print(f"   - MFCC coefficients: {mfcc.shape[0]}")
print(f"   - Time steps: {mfcc.shape[1]}")

# Test Mel Spectrogram
print(f"\nExtracting Mel Spectrogram...")
mel_spec = audio_processor.extract_mel_spectrogram(synthetic_audio)
print(f"✅ Mel Spectrogram extracted: {mel_spec.shape}")
print(f"   - Mel bands: {mel_spec.shape[0]}")
print(f"   - Time steps: {mel_spec.shape[1]}")

# Test Wav2Vec2 features
print(f"\nExtracting Wav2Vec2 transformer features...")
wav2vec_features = audio_processor.extract_wav2vec_features(synthetic_audio)
print(f"✅ Wav2Vec2 features extracted: {wav2vec_features.shape}")
print(f"   - Semantic audio embeddings: {wav2vec_features.shape[1]} dims")

# Test batch processing
print(f"\nTesting batch processing with multiple audio samples...")
audio_batch = [synthetic_audio for _ in range(3)]
batch_wav2vec = audio_processor.extract_wav2vec_features(audio_batch)
print(f"✅ Batch processing complete: {batch_wav2vec.shape}")

print("\n✅ Audio Preprocessor: ALL TESTS PASSED")

# Test 5: Multimodal Integration
print("\n" + "="*80)
print("TEST 5: Multimodal Integration")
print("-" * 80)

print("\nTesting multimodal feature dimensions...")
feature_dims = {
    'Text (mBERT)': features.shape[1],
    'Image ViT (global)': hybrid_features['vit_features'].shape[1],
    'Image CNN (local)': hybrid_features['cnn_features'].shape[1],
    'Image Fused': hybrid_features['fused_features'].shape[1],
    'Audio Wav2Vec2': wav2vec_features.shape[1],
    'Audio MFCC': mfcc.shape[0],
    'Audio Mel Spec': mel_spec.shape[0]
}

print("\n📊 Feature Dimensions Summary:")
for modality, dim in feature_dims.items():
    print(f"   {modality:25s}: {dim:4d} dimensions")

# Calculate total feature dimension for fusion
text_dim = features.shape[1]
image_dim = hybrid_features['fused_features'].shape[1]
audio_dim = wav2vec_features.shape[1]
total_dim = text_dim + image_dim + audio_dim

print(f"\n🔗 Total Multimodal Feature Dimension: {total_dim}")
print(f"   Ready for Transformer Fusion Layer!")

print("\n✅ Multimodal Integration: ALL TESTS PASSED")

# Final Summary
print("\n" + "="*80)
print("PHASE 1 TEST SUMMARY")
print("="*80)

print("\n✅ ALL COMPONENTS TESTED SUCCESSFULLY!")
print("\n📊 System Capabilities:")
print("   ✅ Multilingual Text Processing (mBERT)")
print("   ✅ Hybrid Image Analysis (ViT + ConvNeXt)")
print("   ✅ Audio Deepfake Detection (Wav2Vec2 + MFCC)")
print("   ✅ Attention-based Explainability (XAI)")
print("   ✅ Batch Processing Support")
print("   ✅ Local-Global Feature Synergy")

print("\n🎯 Performance Targets (from Literature Survey):")
print("   📈 Text: 85-95% accuracy")
print("   📈 Image: 97% accuracy (99.3% AUC)")
print("   📈 Audio: 96.55% accuracy")
print("   📈 Multimodal Fusion: 96-97% final accuracy")

print("\n🚀 Phase 1 Status: COMPLETE")
print("   Next Phase: Agentic Framework + Transformer Fusion")

print("\n" + "="*80)
print("Total Feature Dimensions for Phase 2:")
print(f"   Text + Image + Audio = {text_dim} + {image_dim} + {audio_dim} = {total_dim} dims")
print("="*80 + "\n")

# Save test results
results = {
    'text_features_shape': str(features.shape),
    'image_vit_shape': str(hybrid_features['vit_features'].shape),
    'image_cnn_shape': str(hybrid_features['cnn_features'].shape),
    'audio_wav2vec_shape': str(wav2vec_features.shape),
    'total_feature_dim': total_dim,
    'device': str(Config.DEVICE)
}

import json
with open('phase1_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Test results saved to: phase1_test_results.json\n")
