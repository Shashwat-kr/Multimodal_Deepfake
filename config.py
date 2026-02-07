
import torch
from pathlib import Path

class Config:
    """Configuration for Multimodal Deepfake Detection System"""

    # Device Configuration
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Model Paths
    BERT_MODEL = 'bert-base-multilingual-cased'  # mBERT for multilingual support
    VIT_MODEL = 'google/vit-base-patch16-224'    # Vision Transformer
    CONVNEXT_MODEL = 'facebook/convnext-base-224' # ConvNeXt for local features
    SWIN_MODEL = 'microsoft/swin-base-patch4-window7-224'  # Swin Transformer
    WAV2VEC_MODEL = 'facebook/wav2vec2-base-960h'  # Audio processing

    # Text Processing
    MAX_TEXT_LENGTH = 512
    TEXT_EMBEDDING_DIM = 768  # BERT output dimension

    # Image Processing
    IMAGE_SIZE = 224
    IMAGE_PATCH_SIZE = 16
    IMAGE_EMBEDDING_DIM = 768  # ViT output dimension

    # Video Processing
    VIDEO_FPS = 30
    MAX_FRAMES = 300  # 10 seconds at 30fps
    FRAME_SAMPLE_RATE = 5  # Sample every 5th frame

    # Audio Processing
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_MAX_LENGTH = 10  # seconds
    N_MFCC = 40
    N_MELS = 128

    # Data Paths
    DATA_DIR = Path('data')
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = Path('models')
    CACHE_DIR = Path('cache')

    # Batch Processing
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Create directories
    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, 
                         cls.PROCESSED_DATA_DIR, cls.MODELS_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directories created successfully!")
        print(f"   Device: {cls.DEVICE}")
        print(f"   BERT Model: {cls.BERT_MODEL}")
        print(f"   ViT Model: {cls.VIT_MODEL}")

if __name__ == "__main__":
    Config.setup_directories()
