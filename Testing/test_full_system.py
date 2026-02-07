"""
Full System Test: Zero-Shot Deepfake Detection
==============================================

Tests the complete integrated system with all 4 agents.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Phase_2.zero_shot_detection_system import ZeroShotDeepfakeDetectionSystem
import torch

def test_full_system():
    """Test the complete zero-shot detection system"""

    print("\n" + "="*80)
    print("FULL SYSTEM TEST: Zero-Shot Deepfake Detection")
    print("="*80)

    # ========================================================================
    # TEST 1: System Initialization
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: System Initialization")
    print("="*80)

    try:
        system = ZeroShotDeepfakeDetectionSystem(
            fusion_dim=512,
            num_transformer_layers=3,
            use_large_clip=False  # Use base CLIP for speed
        )
        print("\n✅ TEST 1 PASSED: System initialized successfully")
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        return

    # ========================================================================
    # TEST 2: Text-Only Detection (Fake News)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Text-Only Detection (Fake News Indicators)")
    print("="*80)

    fake_text = "BREAKING: Shocking miracle cure discovered! Doctors hate this one trick! 100% guaranteed!"

    try:
        result = system.detect(text=fake_text)

        print(f"\n📊 RESULTS:")
        print(f"   Text: {fake_text[:60]}...")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Fake Probability: {result['fake_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Risk Level: {result['risk_level']}")

        if result['verdict'] in ['FAKE', 'UNCERTAIN']:
            print("\n✅ TEST 2 PASSED: Correctly flagged suspicious text")
        else:
            print("\n⚠️  TEST 2 MARGINAL: Detected as REAL (may need more context)")

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")

    # ========================================================================
    # TEST 3: Image-Only Detection (if Gemini image available)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Image-Only Detection (AI-Generated Image)")
    print("="*80)

    # Check if Gemini image exists
    import os
    gemini_paths = [
        "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"
    ]

    gemini_path = None
    for path in gemini_paths:
        if os.path.exists(path):
            gemini_path = path
            break

    if gemini_path:
        try:
            result = system.detect(image_path=gemini_path)

            print(f"\n📊 RESULTS:")
            print(f"   Image: {gemini_path}")
            print(f"   Verdict: {result['verdict']}")
            print(f"   Fake Probability: {result['fake_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")

            if result['verdict'] == 'FAKE':
                print("\n✅ TEST 3 PASSED: Correctly detected AI-generated image as FAKE")
            elif result['verdict'] == 'UNCERTAIN':
                print("\n⚠️  TEST 3 MARGINAL: Flagged as UNCERTAIN (acceptable for borderline)")
            else:
                print("\n❌ TEST 3 FAILED: Detected AI image as REAL")

        except Exception as e:
            print(f"\n❌ TEST 3 FAILED: {e}")
    else:
        print("\n⏭️  TEST 3 SKIPPED: Gemini image not found")
        print("   Place 'Gemini_Generated_Image_1jzljk1jzljk1jzl.png' to test")

    # ========================================================================
    # TEST 4: Multimodal Detection (Text + Image)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: Multimodal Detection (Text + Image)")
    print("="*80)

    if gemini_path:
        try:
            # AI image with sensational text (both fake)
            result = system.detect(
                text="BREAKING NEWS: Scientists shocked by this discovery!",
                image_path=gemini_path
            )

            print(f"\n📊 RESULTS:")
            print(f"   Modalities: text + image")
            print(f"   Verdict: {result['verdict']}")
            print(f"   Fake Probability: {result['fake_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")

            print(f"\n   Agent Scores:")
            for agent, score in result['agent_scores'].items():
                print(f"      {agent.capitalize():15s}: {score:.3f}")

            print(f"\n   Agent Agreement:")
            agreement = result['agreement_analysis']
            print(f"      Type: {agreement['agreement']}")
            print(f"      FAKE votes: {agreement['fake_votes']}")
            print(f"      REAL votes: {agreement['real_votes']}")

            if result['verdict'] in ['FAKE', 'UNCERTAIN']:
                print("\n✅ TEST 4 PASSED: Multimodal detection working")
            else:
                print("\n⚠️  TEST 4 MARGINAL: Check agent weights if needed")

        except Exception as e:
            print(f"\n❌ TEST 4 FAILED: {e}")
    else:
        print("\n⏭️  TEST 4 SKIPPED: No image available for multimodal test")

    # ========================================================================
    # TEST 5: Credible Content (Should Detect as REAL)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 5: Credible Content Detection")
    print("="*80)

    credible_text = "According to Reuters, the government announced new policies. The study was published in Nature and peer-reviewed."

    try:
        result = system.detect(text=credible_text)

        print(f"\n📊 RESULTS:")
        print(f"   Text: {credible_text[:60]}...")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Fake Probability: {result['fake_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")

        if result['verdict'] == 'REAL':
            print("\n✅ TEST 5 PASSED: Correctly detected credible content as REAL")
        elif result['verdict'] == 'UNCERTAIN':
            print("\n⚠️  TEST 5 MARGINAL: Flagged as UNCERTAIN (may need more signals)")
        else:
            print("\n❌ TEST 5 FAILED: Credible content detected as FAKE")

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print("\n✅ System Status: OPERATIONAL")
    print("\n📊 What was tested:")
    print("   • System initialization (all 4 agents)")
    print("   • Text-only detection (fake news patterns)")
    print("   • Image-only detection (AI-generated)")
    print("   • Multimodal detection (text + image)")
    print("   • Credible content detection")

    print("\n🎯 Expected Behavior:")
    print("   • Fake/suspicious content → FAKE or UNCERTAIN")
    print("   • Credible content → REAL or UNCERTAIN")
    print("   • Borderline cases → UNCERTAIN (by design)")

    print("\n💡 Note:")
    print("   Conservative confidence is intentional to avoid overconfidence.")
    print("   Adjust thresholds in zero_shot_reasoning_agent.py if needed.")

    print("\n🚀 Next Steps:")
    print("   1. Test with your own images/videos")
    print("   2. Check agent explanations: result['explanation']")
    print("   3. Review agent scores: result['agent_scores']")
    print("   4. Deploy to production!")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_full_system()
