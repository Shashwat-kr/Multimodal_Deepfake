
'''
Simple Test with Real Files
Test the deepfake detection system with your own data
'''

import sys
from pathlib import Path
# Add project root (DeepFake/) to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import torch
print("\n" + "="*80)
print("DEEPFAKE DETECTION - SIMPLE TEST WITH REAL FILES")
print("="*80 + "\n")

# Import the complete system
try:
    from Phase_2.detection_system import DeepfakeDetectionSystem
    from Phase_2.explainability import ExplainabilityModule
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("\nMake sure all Phase 1 and Phase 2 files are in the same directory.")
    sys.exit(1)

# Initialize system
print("Initializing detection system...")
print("(This may take a minute as models are loaded)\n")

try:
    system = DeepfakeDetectionSystem()
    explainer = ExplainabilityModule()
    print("✅ System initialized successfully!\n")
except Exception as e:
    print(f"❌ Error initializing system: {e}")
    sys.exit(1)

# ============================================================================
# EXAMPLE 1: TEXT ONLY
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Text-Only Misinformation Detection")
print("="*80 + "\n")

# Test with suspicious text
suspicious_text = """
BREAKING NEWS: Scientists have discovered a miracle cure that works 100% of the time!
Doctors don't want you to know about this secret treatment that Big Pharma is trying to hide.
Share this immediately before it gets taken down! This shocking discovery will change everything!
"""

print("📝 Analyzing text:")
print("-" * 80)
print(suspicious_text.strip())
print("-" * 80)

try:
    result = system.detect(text=suspicious_text)

    print(f"\n🎯 RESULT:")
    print(f"   Verdict:           {result['verdict']}")
    print(f"   Fake Probability:  {result['fake_probability']:.1%}")
    print(f"   Confidence:        {result['confidence']:.1%}")
    print(f"   Risk Level:        {result['risk_level']}")
    print(f"   Modalities:        {', '.join(result['modalities_analyzed'])}")

    if 'modality_importance' in result and result['modality_importance'] is not None:
        print(f"\n📊 Modality Importance:")
        for mod, score in result['modality_importance'].items():
            print(f"      {mod.capitalize():10s}: {score:.4f}")

    print(f"\n💭 Reasoning:")
    print(result['explanation'])

    # Save report
    report_path = 'reports/text_only_report.html'
    Path('reports').mkdir(exist_ok=True)
    explainer.generate_report(result, save_path=report_path)
    print(f"\n✅ HTML report saved to: {report_path}")

except Exception as e:
    print(f"❌ Error during detection: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# EXAMPLE 2: TEXT + IMAGE (if you have an image)
# ============================================================================
print("\n\n" + "="*80)
print("EXAMPLE 2: Text + Image Detection")
print("="*80 + "\n")

print("📝 Instructions:")
print("   To test with your own image, replace the image_path below")
print("   Supported formats: .jpg, .jpeg, .png, .bmp\n")

# You can replace this with your actual image path
image_path = "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"  # Set to your image path, e.g., "path/to/image.jpg"

if image_path and Path(image_path).exists():
    print(f"📷 Testing with image: {image_path}\n")

    test_text = "This amazing photo shows something incredible!"

    try:
        result = system.detect(
            text=test_text,
            image_path=image_path
        )

        print(f"🎯 RESULT:")
        print(f"   Verdict:           {result['verdict']}")
        print(f"   Fake Probability:  {result['fake_probability']:.1%}")
        print(f"   Confidence:        {result['confidence']:.1%}")
        print(f"   Risk Level:        {result['risk_level']}")
        print(f"   Modalities:        {', '.join(result['modalities_analyzed'])}")

        # Save report
        report_path = 'reports/text_image_report.html'
        explainer.generate_report(result, save_path=report_path)
        print(f"\n✅ HTML report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
else:
    print("ℹ️  No image provided. Skipping image test.")
    print("   To test with an image, set: image_path = 'your/image/path.jpg'\n")

# ============================================================================
# EXAMPLE 3: CUSTOM TEST - Edit this section with your files
# ============================================================================
print("\n\n" + "="*80)
print("EXAMPLE 3: CUSTOM TEST (Edit this section)")
print("="*80 + "\n")

print("📝 To test with your own files, edit this section in the script:\n")
print("   text:       Your text content")
print("   image_path: Path to your image file")
print("   audio_path: Path to your audio file (.wav, .mp3)")
print("   video_path: Path to your video file (.mp4, .avi)\n")

# ============================================================================
# EDIT THESE VARIABLES WITH YOUR FILES
# ============================================================================
custom_text = None  # "Your text here" or None
custom_image = None  # "path/to/image.jpg" or None
custom_audio = "Inputs/file_example_WAV_1MG.wav"  # "path/to/audio.wav" or None
custom_video = "Inputs/A.mp4"  # "path/to/video.mp4" or None

# Check if any custom input is provided
has_custom_input = any([custom_text, custom_image, custom_audio, custom_video])

if has_custom_input:
    print("🔍 Running custom detection...\n")

    try:
        result = system.detect(
            text=custom_text,
            image_path=custom_image,
            audio_path=custom_audio,
            video_path=custom_video
        )

        print(f"\n🎯 RESULT:")
        print(f"   Verdict:           {result['verdict']}")
        print(f"   Fake Probability:  {result['fake_probability']:.1%}")
        print(f"   Confidence:        {result['confidence']:.1%}")
        print(f"   Risk Level:        {result['risk_level']}")
        print(f"   Modalities:        {', '.join(result['modalities_analyzed'])}")

        if 'modality_importance' in result and result['modality_importance'] is not None:
            print(f"\n📊 Modality Importance:")
            importance = result.get('modality_importance')
            if importance is not None:
                print("\n📊 Modality Importance:")
                if isinstance(importance, dict):
                    # Already in correct format
                    for mod, score in importance.items():
                        print(f"      {mod.capitalize():10s}: {float(score):.4f}")

                elif torch.is_tensor(importance):
                    # Tensor → map to modality names
                    modality_names = result.get('modalities_analyzed', [])
                    for mod, score in zip(modality_names, importance.tolist()):
                        # score may be float, tensor, or list → normalize it
                        if isinstance(score, list):
                            score = score[0] if len(score) > 0 else 0.0

                        print(f"      {mod.capitalize():10s}: {float(score):.4f}")

        print(f"\n💭 Reasoning:")
        print(result['explanation'])

        # Save report
        report_path = 'reports/custom_report.html'
        explainer.generate_report(result, save_path=report_path)
        print(f"\n✅ HTML report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
else:
    print("ℹ️  No custom input provided.")
    print("   Edit the variables above to test with your own files.\n")

# ============================================================================
# INTERACTIVE MODE (Optional)
# ============================================================================
print("\n\n" + "="*80)
print("INTERACTIVE MODE")
print("="*80 + "\n")

print("Want to test with your own text right now?\n")

try:
    user_input = input("Enter suspicious text (or press Enter to skip): ").strip()

    if user_input:
        print("\n🔍 Analyzing your input...\n")

        result = system.detect(text=user_input)

        print(f"🎯 RESULT:")
        print(f"   Verdict:           {result['verdict']}")
        print(f"   Fake Probability:  {result['fake_probability']:.1%}")
        print(f"   Confidence:        {result['confidence']:.1%}")
        print(f"   Risk Level:        {result['risk_level']}")

        # Save report
        report_path = 'reports/interactive_report.html'
        explainer.generate_report(result, save_path=report_path)
        print(f"\n✅ HTML report saved to: {report_path}")
    else:
        print("Skipped interactive mode.")

except KeyboardInterrupt:
    print("\n\nInterrupted by user.")
except Exception as e:
    print(f"\n❌ Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("TEST SUMMARY")
print("="*80 + "\n")

print("✅ Testing complete!")
print("\n📊 System Capabilities:")
print("   • Text misinformation detection (multilingual)")
print("   • Image deepfake detection")
print("   • Audio deepfake detection")
print("   • Video deepfake detection")
print("   • Multimodal fusion & reasoning")
print("   • Explainable outputs with Chain-of-Thought")

print("\n📁 Reports saved in: ./reports/")
print("   Open the HTML files in a browser to view detailed analysis")

print("\n💡 Next Steps:")
print("   1. Edit this script to add your own files")
print("   2. Check the generated HTML reports")
print("   3. Try different combinations of modalities")

print("\n" + "="*80 + "\n")
