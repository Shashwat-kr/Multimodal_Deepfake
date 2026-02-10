import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from PIL import Image
from Phase_2.zero_shot_visual_detector import EnhancedZeroShotVisualDetectorV2

# Initialize detector
detector = EnhancedZeroShotVisualDetectorV2()

# Load image (change path!)
image_path = "Inputs/Gemini_Generated_Image_1jzljk1jzljk1jzl.png"
image = Image.open(image_path).convert("RGB")

# Run analysis
result = detector.analyze_from_image(image)

# Print results
print("\n===== DETECTION RESULT =====")
print(f"Verdict    : {result.verdict}")
print(f"Score      : {result.score:.3f}")
print(f"Confidence : {result.confidence:.2f}")
print("\nReasoning:")
print(result.reasoning)

print("\nEvidence:")
for e in result.evidence:
    print("-", e)