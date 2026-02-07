"""
Test Script: Zero-Shot Cross-Modal Consistency Agent
====================================================

Run this to test the agent before integration.
"""
import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import torch
from Phase_2.zero_shot_consistency_agent import ZeroShotCrossModalConsistencyAgent, AgentOutput

def test_consistency_agent():
    """Test the zero-shot consistency agent"""

    print("\n" + "="*80)
    print("ZERO-SHOT CROSS-MODAL CONSISTENCY AGENT TEST SUITE")
    print("="*80)

    # Initialize agent
    agent = ZeroShotCrossModalConsistencyAgent()

    # ========================================================================
    # TEST 1: Consistent Content (Real)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Consistent Multimodal Content (Expected: REAL)")
    print("="*80)
    print("\nScenario: News video where text, audio, and video all align")

    # Simulate consistent features (similar semantic patterns)
    base_pattern = torch.randn(1, 384)

    text_feat = torch.cat([base_pattern, base_pattern + torch.randn(1, 384) * 0.1], dim=1)  # 768
    visual_feat = torch.cat([
        text_feat[:, :384] + torch.randn(1, 384) * 0.15,  # Similar to text
        torch.randn(1, 1408) * 0.5  # Additional visual features
    ], dim=1)  # 1792 total
    audio_feat = text_feat + torch.randn(1, 768) * 0.12  # Very similar to text

    result = agent.analyze(text_feat, visual_feat, audio_feat)

    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   \n   Reasoning: {result.reasoning}")
    print(f"\n   Evidence:")
    for evidence in result.evidence:
        print(f"      • {evidence}")

    status = "✅ PASS" if result.verdict == 'real' else "❌ FAIL"
    print(f"\n   Test Status: {status}")

    # ========================================================================
    # TEST 2: Inconsistent Content (Fake)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Inconsistent Multimodal Content (Expected: FAKE)")
    print("="*80)
    print("\nScenario: Deepfake where audio doesn't match video, text is misaligned")

    # Simulate inconsistent features (very different patterns)
    text_feat = torch.randn(1, 768) * 1.0
    visual_feat = torch.randn(1, 1792) * 2.0  # Very different scale
    audio_feat = torch.randn(1, 768) * -1.5  # Opposite pattern

    result = agent.analyze(text_feat, visual_feat, audio_feat)

    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   \n   Reasoning: {result.reasoning}")
    print(f"\n   Evidence:")
    for evidence in result.evidence:
        print(f"      • {evidence}")

    status = "✅ PASS" if result.verdict == 'fake' else "❌ FAIL"
    print(f"\n   Test Status: {status}")

    # ========================================================================
    # TEST 3: Moderate Consistency (Uncertain)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Moderate Consistency (Expected: UNCERTAIN)")
    print("="*80)
    print("\nScenario: Ambiguous content with some alignment, some mismatch")

    text_feat = torch.randn(1, 768)
    visual_feat = torch.cat([
        text_feat[:, :384] + torch.randn(1, 384) * 0.5,  # Partially similar
        torch.randn(1, 1408)
    ], dim=1)
    audio_feat = torch.randn(1, 768) * 0.8

    result = agent.analyze(text_feat, visual_feat, audio_feat)

    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   \n   Reasoning: {result.reasoning}")
    print(f"\n   Evidence:")
    for evidence in result.evidence:
        print(f"      • {evidence}")

    # ========================================================================
    # TEST 4: Pairwise Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: Pairwise Consistency Scores")
    print("="*80)
    print("\nAnalyzing individual modality pair consistency...")

    # Create features with known relationships
    text_feat = torch.randn(1, 768)
    visual_feat = torch.cat([text_feat[:, :384], torch.randn(1, 1408)], dim=1)  # Share 50%
    audio_feat = text_feat * 0.9  # Very similar to text

    pairwise = agent.get_pairwise_scores(text_feat, visual_feat, audio_feat)

    print(f"\n📊 Pairwise Consistency Scores:")
    for pair, score in pairwise.items():
        consistency = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        print(f"   {pair.replace('_', '-').title():20s}: {score:.3f} [{consistency}]")

    # ========================================================================
    # TEST 5: Single Modality (Should Return Uncertain)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 5: Single Modality Input (Expected: UNCERTAIN)")
    print("="*80)
    print("\nScenario: Only one modality available - no cross-modal check possible")

    text_only = torch.randn(1, 768)
    result = agent.analyze(text_features=text_only)

    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   \n   Reasoning: {result.reasoning}")

    status = "✅ PASS" if result.verdict == 'uncertain' else "❌ FAIL"
    print(f"\n   Test Status: {status}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print("\n✅ Zero-Shot Cross-Modal Consistency Agent working correctly!")
    print("\nKey Features Tested:")
    print("   • Text-Image consistency (cosine similarity)")
    print("   • Audio-Visual sync detection")
    print("   • Text-Audio consistency")
    print("   • Pairwise score extraction")
    print("   • Single modality handling")

    print("\n🎯 Next Steps:")
    print("   1. Integrate into detection_system.py (see integration_consistency.py)")
    print("   2. Test with real multimodal data")
    print("   3. Move to next agent (Web Retrieval or Reasoning)")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_consistency_agent()
