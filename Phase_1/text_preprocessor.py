
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Union
import numpy as np
from config import Config

class TextPreprocessor:
    """
    Text preprocessing using mBERT (Multilingual BERT)
    Supports Hindi, Gujarati, Marathi, Telugu, and English
    Reference: HEMT-Fake architecture from literature survey
    """

    def __init__(self, model_name: str = Config.BERT_MODEL):
        self.device = Config.DEVICE
        self.max_length = Config.MAX_TEXT_LENGTH

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager"
        ).to(self.device)
        self.model.eval()

        print(f"✅ Text Preprocessor initialized on {self.device}")
        print(f"   Max Length: {self.max_length}")
        print(f"   Embedding Dim: {Config.TEXT_EMBEDDING_DIM}")

    def preprocess_text(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode text using mBERT tokenizer

        Args:
            text: Single string or list of strings

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        return encoded

    def extract_features(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract BERT embeddings (768-dim) from text

        Args:
            text: Single string or list of strings

        Returns:
            Tensor of shape (batch_size, 768) - CLS token embeddings
        """
        with torch.no_grad():
            encoded = self.preprocess_text(text)
            outputs = self.model(**encoded)

            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Optional: Use mean pooling instead
            # attention_mask = encoded['attention_mask'].unsqueeze(-1)
            # mean_embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

        return cls_embeddings

    def extract_features_with_attention(self, text: Union[str, List[str]]) -> Dict:
        """
        Extract features along with attention weights for explainability

        Returns:
            Dictionary with embeddings and attention weights
        """
        with torch.no_grad():
            encoded = self.preprocess_text(text)
            outputs = self.model(**encoded, output_attentions=True)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            attention_weights = outputs.attentions  # List of attention matrices

        return {
            'embeddings': cls_embeddings,
            'attention_weights': attention_weights,
            'tokens': self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        }

    def batch_process(self, texts: List[str], batch_size: int = Config.BATCH_SIZE) -> torch.Tensor:
        """
        Process large list of texts in batches

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Tensor of shape (len(texts), 768)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.extract_features(batch)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# Test the preprocessor
if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()

    # Test with multilingual samples
    test_texts = [
        "This is a fake news article about politics.",
        "यह एक नकली समाचार है।",  # Hindi
        "આ એક બનાવટી સમાચાર છે."  # Gujarati
    ]

    print("\n" + "="*60)
    print("Testing Text Preprocessor")
    print("="*60)

    # Extract features
    features = preprocessor.extract_features(test_texts)
    print(f"\n✅ Feature Extraction Complete:")
    print(f"   Input texts: {len(test_texts)}")
    print(f"   Output shape: {features.shape}")
    print(f"   Embedding dim: {features.shape[1]}")

    # Test attention extraction for explainability
    result = preprocessor.extract_features_with_attention([test_texts[0]])
    print(f"\n✅ Attention Extraction Complete:")
    print(f"   Number of attention layers: {len(result['attention_weights'])}")
    print(f"   Tokens: {result['tokens'][:10]}...")
