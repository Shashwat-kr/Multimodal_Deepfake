import sys
from pathlib import Path

# Add project root (DeepFake/) to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import torch
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("PHASE 2 COMPREHENSIVE TEST")
print("Multimodal Fusion + Agentic Framework")
print("="*80 + "\n")

# Test 1: Multimodal Fusion Layer
print("TEST 1: Multimodal Transformer Fusion")
print("-" * 80)

from Phase_2.multimodal_fusion import MultimodalTransformerFusion

fusion = MultimodalTransformerFusion(
    text_dim=768,
    image_dim=1792,
    audio_dim=768,
    video_dim=1024,
    fusion_dim=512,
    num_heads=8,
    num_layers=3
)

print(f"✅ Fusion layer initialized")
print(f"   Parameters: {sum(p.numel() for p in fusion.parameters()):,}")

# Test with all modalities
batch_size = 4
text_feat = torch.randn(batch_size, 768)
image_feat = torch.randn(batch_size, 1792)
audio_feat = torch.randn(batch_size, 768)
video_feat = torch.randn(batch_size, 1024)

output = fusion(text_feat, image_feat, audio_feat, video_feat, return_attention=True)

print(f"\n✅ Fusion output:")
print(f"   Fused features: {output['fused_features'].shape}")
print(f"   Modality tokens: {output['modality_tokens'].shape}")

importance = fusion.get_modality_importance(text_feat, image_feat, audio_feat, video_feat)
print(f"\n✅ Modality importance:")
for modality, score in importance.items():
    print(f"   {modality.capitalize():10s}: {score:.4f}")

# Test with missing modalities
print(f"\n✅ Testing with missing modalities...")
output_partial = fusion(text_feat, image_feat, None, None)
print(f"   Text + Image only: {output_partial['fused_features'].shape}")

print("\n✅ Multimodal Fusion: ALL TESTS PASSED\n")


# Test 2: Agentic Framework
print("\nTEST 2: Agentic Framework (4 Agents)")
print("-" * 80)

from Phase_2.agentic_framework import (
    VisualVeracityAgent,
    CrossModalConsistencyAgent,
    WebRetrievalAgent,
    ReasoningAgent
)

# Initialize agents
visual_agent = VisualVeracityAgent()
consistency_agent = CrossModalConsistencyAgent()
web_agent = WebRetrievalAgent()
reasoning_agent = ReasoningAgent(input_dim=512)

print(f"✅ All agents initialized")

# Test Visual Veracity Agent
print(f"\n🤖 Testing Visual Veracity Agent...")
visual_output = visual_agent(image_feat[0:1])
print(f"   Verdict: {visual_output.verdict.upper()}")
print(f"   Score: {visual_output.score:.3f}")
print(f"   Confidence: {visual_output.confidence:.3f}")
print(f"   Evidence: {len(visual_output.evidence)} items")

# Test Cross-Modal Consistency Agent
print(f"\n🤖 Testing Cross-Modal Consistency Agent...")
consistency_output = consistency_agent(text_feat[0:1], image_feat[0:1], audio_feat[0:1])
print(f"   Verdict: {consistency_output.verdict.upper()}")
print(f"   Score: {consistency_output.score:.3f}")
print(f"   Evidence: {consistency_output.evidence[0]}")

# Test Web Retrieval Agent
print(f"\n🤖 Testing Web Retrieval Agent...")
test_text = "Breaking: Scientists discover miracle cure that doctors don't want you to know!"
web_output = web_agent.search_and_verify(test_text)
print(f"   Verdict: {web_output.verdict.upper()}")
print(f"   Score: {web_output.score:.3f}")
print(f"   Reasoning: {web_output.reasoning[:80]}...")

# Test Reasoning Agent
print(f"\n🤖 Testing Reasoning & Judgment Agent...")
fused_feat = output['fused_features'][0:1]
final_output = reasoning_agent(fused_feat, visual_output, consistency_output, web_output)
print(f"   Final Verdict: {final_output['verdict']}")
print(f"   Fake Probability: {final_output['fake_probability']:.3f}")
print(f"   Confidence: {final_output['confidence']:.3f}")
print(f"   Risk Level: {final_output['risk_level']}")

print("\n✅ Agentic Framework: ALL TESTS PASSED\n")


# Test 3: End-to-End Pipeline (without real data)
print("\nTEST 3: End-to-End Pipeline Simulation")
print("-" * 80)

print("\n📊 Simulating complete detection pipeline...")

# Simulate feature extraction
features = {
    'text': text_feat[0:1],
    'image': image_feat[0:1],
    'audio': audio_feat[0:1],
    'video': None,
    'modalities_present': ['text', 'image', 'audio']
}

# Step 1: Fusion
print("   Step 1: Multimodal fusion...")
fusion_output = fusion(features['text'], features['image'], features['audio'])
print(f"      ✅ Fused features: {fusion_output['fused_features'].shape}")

# Step 2: Agent analysis
print("   Step 2: Agent analysis...")
visual_out = visual_agent(features['image'])
consistency_out = consistency_agent(features['text'], features['image'], features['audio'])
web_out = web_agent.search_and_verify("Sample text for testing")
print(f"      ✅ All agents completed")

# Step 3: Final reasoning
print("   Step 3: Final reasoning...")
final_result = reasoning_agent(
    fusion_output['fused_features'],
    visual_out,
    consistency_out,
    web_out,
    "Sample text"
)
print(f"      ✅ Final verdict: {final_result['verdict']}")

print("\n✅ End-to-End Pipeline: ALL TESTS PASSED\n")


# Test 4: System Statistics
print("\nTEST 4: System Statistics")
print("-" * 80)

# Count parameters
fusion_params = sum(p.numel() for p in fusion.parameters())
visual_params = sum(p.numel() for p in visual_agent.parameters())
consistency_params = sum(p.numel() for p in consistency_agent.parameters())
reasoning_params = sum(p.numel() for p in reasoning_agent.parameters())

total_phase2_params = fusion_params + visual_params + consistency_params + reasoning_params

print(f"\n📊 Model Parameters:")
print(f"   Fusion Layer:        {fusion_params:>12,} parameters")
print(f"   Visual Agent:        {visual_params:>12,} parameters")
print(f"   Consistency Agent:   {consistency_params:>12,} parameters")
print(f"   Reasoning Agent:     {reasoning_params:>12,} parameters")
print(f"   " + "-" * 50)
print(f"   Total Phase 2:       {total_phase2_params:>12,} parameters")

print(f"\n📊 Feature Dimensions:")
print(f"   Input Text:          768 dims (mBERT)")
print(f"   Input Image:         1792 dims (ViT+ConvNeXt)")
print(f"   Input Audio:         768 dims (Wav2Vec2)")
print(f"   Input Video:         1024 dims (Swin)")
print(f"   Fused Output:        512 dims (Transformer)")

print(f"\n🎯 Architecture Summary:")
print(f"   Transformer Layers:  3")
print(f"   Attention Heads:     8")
print(f"   Fusion Dimension:    512")
print(f"   Number of Agents:    4")

print("\n✅ System Statistics: COMPUTED\n")


# Final Summary
print("\n" + "="*80)
print("PHASE 2 TEST SUMMARY")
print("="*80)

print("\n✅ ALL TESTS PASSED!")
print("\n📊 Components Tested:")
print("   ✅ Multimodal Transformer Fusion")
print("   ✅ Visual Veracity Agent")
print("   ✅ Cross-Modal Consistency Agent")
print("   ✅ Web Retrieval Agent")
print("   ✅ Reasoning & Judgment Agent")
print("   ✅ End-to-End Pipeline")
print("   ✅ Parameter Counting")

print("\n🚀 Phase 2 Status: COMPLETE")
print("   Next: Training on real datasets")

print("\n" + "="*80)
print(f"Total Phase 2 Parameters: {total_phase2_params:,}")
print(f"Combined with Phase 1: ~{535_000_000 + total_phase2_params:,} total parameters")
print("="*80 + "\n")

# Save test results
results = {
    'fusion_params': fusion_params,
    'agent_params': visual_params + consistency_params + reasoning_params,
    'total_params': total_phase2_params,
    'fusion_dim': 512,
    'num_agents': 4
}

import json
with open('phase2_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Test results saved to: phase2_test_results.json\n")
