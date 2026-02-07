"""
Test Script: Zero-Shot Visual Veracity Agent
=============================================

Run this to test the agent on sample images before integrating.

Requirements:
    pip install torch transformers pillow
"""
import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import torch
from Phase_2.zero_shot_visual_agent import ZeroShotVisualVeracityAgent, AgentOutput
from PIL import Image
import urllib.request
import os

def download_test_image(url: str, filename: str):
    """Download a test image if not already present"""
    if not os.path.exists(filename):
        print(f"📥 Downloading test image: {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded: {filename}")
    else:
        print(f"✅ Using existing: {filename}")
    return filename


def test_zero_shot_visual_agent():
    """
    Test the zero-shot visual agent with sample images
    """
    print("\n" + "="*80)
    print("ZERO-SHOT VISUAL AGENT TEST SUITE")
    print("="*80 + "\n")

    # Initialize agent
    print("Step 1: Initializing Zero-Shot Visual Agent...")
    agent = ZeroShotVisualVeracityAgent(
        model_name="openai/clip-vit-base-patch32",  # Use base for faster testing
        device="mps"  # Change to "cuda" or "mps" if available
    )

    # Test cases
    print("\nStep 2: Preparing test images...")

    test_cases = []

    # You can use your own images or download some
    # For now, let's test with any image you have

    # Example 1: If you have the image from your files
    if os.path.exists("Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"):
        test_cases.append({
            'path': "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png",
            'description': "Gemini Generated Image (AI-generated)",
            'expected': 'fake'
        })

    # Example 2: Create a synthetic test case
    print("\n📸 Test Images:")
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case['description']}")

    if not test_cases:
        print("\n⚠️  No test images found!")
        print("\nPlease add images to test:")
        print("   • Place real photos in the current directory")
        print("   • Or specify image paths in this script")
        print("\nFor now, demonstrating with feature-based analysis...")

        # Test with dummy features
        print("\n" + "="*80)
        print("TEST: Feature-based Analysis (Fallback Mode)")
        print("="*80)

        dummy_features = torch.randn(1, 1792)
        result = agent.analyze_from_features(dummy_features)

        print(f"\n📊 Result:")
        print(f"   Verdict: {result.verdict.upper()}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Reasoning: {result.reasoning}")
        print(f"\n   Evidence:")
        for evidence in result.evidence:
            print(f"      • {evidence}")

        return

    # Run tests
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)

    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {case['description']}")
        print(f"{'='*80}")

        try:
            # Load image
            image = Image.open(case['path']).convert('RGB')
            print(f"✅ Loaded image: {image.size}")

            # Analyze
            print("\n🔍 Analyzing with CLIP zero-shot detection...")
            result = agent.analyze_from_image(image)

            # Display results
            print(f"\n📊 RESULT:")
            print(f"   Verdict: {result.verdict.upper()}")
            print(f"   Score: {result.score:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Expected: {case['expected'].upper()}")

            # Check if correct
            is_correct = result.verdict == case['expected']
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"   Status: {status}")

            print(f"\n   Reasoning: {result.reasoning}")

            print(f"\n   Evidence:")
            for evidence in result.evidence:
                print(f"      • {evidence}")

            results.append({
                'case': case['description'],
                'result': result,
                'correct': is_correct
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = (correct / total) * 100

        print(f"\n✅ Passed: {correct}/{total} ({accuracy:.1f}%)")
        print(f"\n📋 Detailed Results:")
        for r in results:
            status = "✅" if r['correct'] else "❌"
            print(f"   {status} {r['case']}: {r['result'].verdict.upper()} "
                  f"(confidence: {r['result'].confidence:.3f})")

    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")


def test_with_custom_image(image_path: str):
    """
    Quick test with your own image

    Args:
        image_path: Path to your image file
    """
    print(f"\n{'='*80}")
    print(f"Testing with custom image: {image_path}")
    print(f"{'='*80}\n")

    # Initialize agent
    agent = ZeroShotVisualVeracityAgent()

    # Load and analyze
    image = Image.open(image_path).convert('RGB')
    result = agent.analyze_from_image(image)

    # Display
    print(f"\n📊 RESULT:")
    print(f"   Verdict: {result.verdict.upper()}")
    print(f"   Authenticity Score: {result.score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"\n   Reasoning: {result.reasoning}")
    print(f"\n   Evidence:")
    for evidence in result.evidence:
        print(f"      • {evidence}")

    return result


def compare_real_vs_fake():
    """
    Side-by-side comparison of real vs fake images
    """
    print(f"\n{'='*80}")
    print("REAL vs FAKE COMPARISON")
    print(f"{'='*80}\n")

    agent = ZeroShotVisualVeracityAgent()

    # You need to provide paths to real and fake images
    real_image_path = "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"  # UPDATE THIS
    fake_image_path = "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"  # UPDATE THIS

    if os.path.exists(real_image_path):
        print("Analyzing REAL image...")
        real_img = Image.open(real_image_path).convert('RGB')
        real_result = agent.analyze_from_image(real_img)
        print(f"   Real Image Score: {real_result.score:.3f} ({real_result.verdict})")

    if os.path.exists(fake_image_path):
        print("\nAnalyzing FAKE image...")
        fake_img = Image.open(fake_image_path).convert('RGB')
        fake_result = agent.analyze_from_image(fake_img)
        print(f"   Fake Image Score: {fake_result.score:.3f} ({fake_result.verdict})")

    if os.path.exists(real_image_path) and os.path.exists(fake_image_path):
        print(f"\n📊 Score Difference: {abs(real_result.score - fake_result.score):.3f}")


if __name__ == "__main__":
    # Run main test suite
    test_zero_shot_visual_agent()

    # Uncomment to test with your own image:
    # test_with_custom_image("path/to/your/image.jpg")

    # Uncomment for side-by-side comparison:
    # compare_real_vs_fake()