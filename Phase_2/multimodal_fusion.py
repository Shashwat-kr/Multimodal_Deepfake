
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from config import Config

class MultimodalTransformerFusion(nn.Module):
    """
    Transformer-based Multimodal Fusion Layer

    Uses cross-attention to fuse text, image, audio, and video features.
    Implements dynamic modality weighting based on attention scores.

    Reference: BERT-1DCCNet (93.12% accuracy) and Transformer LLM approaches
    from literature survey achieving 96-97% multimodal accuracy.
    """

    def __init__(self,
                 text_dim: int = 768,
                 image_dim: int = 1792,  # ViT + ConvNeXt fused
                 audio_dim: int = 768,
                 video_dim: int = 1024,
                 fusion_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            text_dim: Dimension of text features (mBERT)
            image_dim: Dimension of image features (ViT+ConvNeXt)
            audio_dim: Dimension of audio features (Wav2Vec2)
            video_dim: Dimension of video features (Swin)
            fusion_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.fusion_dim = fusion_dim

        # Project each modality to common fusion dimension
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.audio_projection = nn.Linear(audio_dim, fusion_dim)
        self.video_projection = nn.Linear(video_dim, fusion_dim)

        # Modality embeddings (learnable positional embeddings for each modality)
        self.text_embedding = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.image_embedding = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.audio_embedding = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.video_embedding = nn.Parameter(torch.randn(1, 1, fusion_dim))

        # Cross-modal transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dynamic modality weighting (attention-based gating)
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None,
                video_features: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-modal fusion

        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
            audio_features: (batch_size, audio_dim)
            video_features: (batch_size, video_dim)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with fused features and optional attention weights
        """
        batch_size = text_features.shape[0] if text_features is not None else                      image_features.shape[0] if image_features is not None else                      audio_features.shape[0] if audio_features is not None else                      video_features.shape[0]

        modality_tokens = []
        modality_masks = []

        # Process text modality
        if text_features is not None:
            text_proj = self.text_projection(text_features).unsqueeze(1)  # (B, 1, D)
            text_proj = text_proj + self.text_embedding
            modality_tokens.append(text_proj)
            modality_masks.append(False)
        else:
            modality_masks.append(True)

        # Process image modality
        if image_features is not None:
            image_proj = self.image_projection(image_features).unsqueeze(1)  # (B, 1, D)
            image_proj = image_proj + self.image_embedding
            modality_tokens.append(image_proj)
            modality_masks.append(False)
        else:
            modality_masks.append(True)

        # Process audio modality
        if audio_features is not None:
            audio_proj = self.audio_projection(audio_features).unsqueeze(1)  # (B, 1, D)
            audio_proj = audio_proj + self.audio_embedding
            modality_tokens.append(audio_proj)
            modality_masks.append(False)
        else:
            modality_masks.append(True)

        # Process video modality
        if video_features is not None:
            video_proj = self.video_projection(video_features).unsqueeze(1)  # (B, 1, D)
            video_proj = video_proj + self.video_embedding
            modality_tokens.append(video_proj)
            modality_masks.append(False)
        else:
            modality_masks.append(True)

        # Concatenate all available modalities
        if len(modality_tokens) == 0:
            raise ValueError("At least one modality must be provided")

        multimodal_tokens = torch.cat(modality_tokens, dim=1)  # (B, num_modalities, D)

        # Apply transformer for cross-modal attention
        fused_features = self.transformer(multimodal_tokens)  # (B, num_modalities, D)

        # Apply modality-specific attention for dynamic weighting
        query = fused_features.mean(dim=1, keepdim=True)  # (B, 1, D)
        attended_features, attention_weights = self.modality_attention(
            query, fused_features, fused_features
        )

        # Final fusion: weighted sum + residual connection
        fused_output = attended_features.squeeze(1)  # (B, D)
        fused_output = self.layer_norm(fused_output + query.squeeze(1))
        fused_output = self.dropout(fused_output)

        result = {
            'fused_features': fused_output,  # (B, fusion_dim)
            'modality_tokens': fused_features,  # (B, num_modalities, D)
        }

        if return_attention:
            result['attention_weights'] = attention_weights  # (B, 1, num_modalities)
            result['modality_importance'] = attention_weights.squeeze(1)  # (B, num_modalities)

        return result

    def get_modality_importance(self, 
                                text_features: Optional[torch.Tensor] = None,
                                image_features: Optional[torch.Tensor] = None,
                                audio_features: Optional[torch.Tensor] = None,
                                video_features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Get importance scores for each modality

        Returns:
            Dictionary with modality names and their importance scores
        """
        with torch.no_grad():
            result = self.forward(
                text_features, image_features, audio_features, video_features,
                return_attention=True
            )

            importance_tensor = result['modality_importance'][0]

            # Ensure float32 for MPS compatibility
            importance_tensor = importance_tensor.to(dtype=torch.float32)

            # Normalize importance scores safely
            scores = torch.softmax(importance_tensor, dim=0)

            modality_names = []
            if text_features is not None:
                modality_names.append('text')
            if image_features is not None:
                modality_names.append('image')
            if audio_features is not None:
                modality_names.append('audio')
            if video_features is not None:
                modality_names.append('video')

            return {
                name: float(score)
                for name, score in zip(modality_names, scores.detach().cpu().tolist())
            }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Multimodal Transformer Fusion")
    print("="*80)

    # Initialize fusion layer
    fusion = MultimodalTransformerFusion(
        text_dim=768,
        image_dim=1792,
        audio_dim=768,
        video_dim=1024,
        fusion_dim=512,
        num_heads=8,
        num_layers=3
    )

    print(f"\n✅ Fusion layer initialized")
    print(f"   Fusion dimension: 512")
    print(f"   Transformer heads: 8")
    print(f"   Transformer layers: 3")

    # Create dummy features
    batch_size = 2
    text_feat = torch.randn(batch_size, 768)
    image_feat = torch.randn(batch_size, 1792)
    audio_feat = torch.randn(batch_size, 768)
    video_feat = torch.randn(batch_size, 1024)

    print(f"\n📊 Input Features:")
    print(f"   Text: {text_feat.shape}")
    print(f"   Image: {image_feat.shape}")
    print(f"   Audio: {audio_feat.shape}")
    print(f"   Video: {video_feat.shape}")

    # Test fusion
    output = fusion(text_feat, image_feat, audio_feat, video_feat, return_attention=True)

    print(f"\n✅ Fusion Output:")
    print(f"   Fused features: {output['fused_features'].shape}")
    print(f"   Modality tokens: {output['modality_tokens'].shape}")

    # Get modality importance
    importance = fusion.get_modality_importance(text_feat, image_feat, audio_feat, video_feat)
    print(f"\n📊 Modality Importance Scores:")
    for modality, score in importance.items():
        print(f"   {modality.capitalize():10s}: {score:.4f}")

    print(f"\n✅ Transformer Fusion Layer: TEST PASSED")
    print("="*80 + "\n")
