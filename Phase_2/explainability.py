
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
import cv2
from PIL import Image

class ExplainabilityModule:
    """
    Explainability Module for Deepfake Detection

    Provides:
    - Grad-CAM visualization for images/videos
    - Attention heatmaps for text
    - Feature importance analysis
    - Chain-of-Thought explanations

    Reference: ViT+CNN+XAI achieving 97.2% with explainability
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')

    def generate_gradcam(self,
                        model: nn.Module,
                        input_tensor: torch.Tensor,
                        target_layer: nn.Module,
                        image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generate Grad-CAM visualization

        Args:
            model: Neural network model
            input_tensor: Input image tensor
            target_layer: Layer to compute gradients for
            image_size: Size to resize heatmap to

        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        model.eval()
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        output = model(input_tensor)

        # Backward pass
        model.zero_grad()
        output.backward(torch.ones_like(output))

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Compute Grad-CAM
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0]
            gradient = gradients[0]

            # Global average pooling of gradients
            weights = gradient.mean(dim=(2, 3), keepdim=True)

            # Weighted combination of activation maps
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # Resize to image size
            cam = F.interpolate(cam, size=image_size, mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            return cam

        return np.zeros(image_size)

    def visualize_gradcam(self,
                         image_path: str,
                         heatmap: np.ndarray,
                         save_path: Optional[str] = None,
                         alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image

        Args:
            image_path: Path to original image
            heatmap: Grad-CAM heatmap
            save_path: Optional path to save visualization
            alpha: Transparency of overlay

        Returns:
            Visualization as numpy array
        """
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))

        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

        if save_path:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return overlay

    def visualize_attention_heatmap(self,
                                   tokens: List[str],
                                   attention_weights: torch.Tensor,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize attention weights for text

        Args:
            tokens: List of tokens
            attention_weights: Attention weight tensor (num_heads, seq_len, seq_len)
            save_path: Optional path to save visualization
        """
        # Average across attention heads
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Take first sample

        avg_attention = attention_weights.mean(dim=0).cpu().numpy()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')

        # Set ticks
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)

        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.title('Attention Heatmap')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def highlight_important_tokens(self,
                                   text: str,
                                   tokens: List[str],
                                   attention_scores: np.ndarray,
                                   top_k: int = 10) -> str:
        """
        Highlight most important tokens in text

        Args:
            text: Original text
            tokens: List of tokens
            attention_scores: Attention scores for each token
            top_k: Number of top tokens to highlight

        Returns:
            HTML string with highlighted text
        """
        # Get top-k most attended tokens
        top_indices = np.argsort(attention_scores)[-top_k:]

        html = '<div style="line-height: 2.5;">'

        for i, token in enumerate(tokens):
            if i in top_indices:
                intensity = attention_scores[i]
                color = f'rgba(255, 0, 0, {intensity:.2f})'
                html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;">{token}</span>'
            else:
                html += f'<span style="margin: 2px;">{token}</span>'

        html += '</div>'
        return html

    def generate_saliency_map(self,
                             model: nn.Module,
                             input_tensor: torch.Tensor,
                             image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generate saliency map using gradients

        Args:
            model: Neural network model
            input_tensor: Input image tensor
            image_size: Size to resize map to

        Returns:
            Saliency map as numpy array
        """
        input_tensor.requires_grad = True

        # Forward pass
        output = model(input_tensor)

        # Backward pass
        model.zero_grad()
        output.backward(torch.ones_like(output))

        # Get gradients
        saliency = input_tensor.grad.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def generate_report(self,
                       detection_result: Dict,
                       save_path: str = 'detection_report.html') -> str:
        """
        Generate comprehensive HTML report

        Args:
            detection_result: Output from detection system
            save_path: Path to save HTML report

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .verdict {{
                    font-size: 48px;
                    font-weight: bold;
                    text-align: center;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    background-color: {'#ff4444' if detection_result['verdict'] == 'FAKE' else '#44ff44'};
                    color: white;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .explanation {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .agent-output {{
                    background: #f9f9f9;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #667eea;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Deepfake Detection Report</h1>
                <p>Multimodal Analysis with Explainable AI</p>
            </div>

            <div class="verdict">
                {detection_result['verdict']}
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{detection_result['fake_probability']:.1%}</div>
                    <div>Fake Probability</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{detection_result['confidence']:.1%}</div>
                    <div>Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{detection_result['risk_level']}</div>
                    <div>Risk Level</div>
                </div>
            </div>

            <div class="explanation">
                <h2>📋 Detailed Explanation</h2>
                <pre style="white-space: pre-wrap; font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 5px;">
{detection_result.get('explanation', 'No explanation available')}
                </pre>
            </div>

            <div class="explanation">
                <h2>🤖 Agent Analysis</h2>
        """

        if 'agent_outputs' in detection_result:
            for agent_name, agent_output in detection_result['agent_outputs'].items():
                html += f"""
                <div class="agent-output">
                    <h3>{agent_name.replace('_', ' ').title()}</h3>
                    <p><strong>Verdict:</strong> {agent_output.verdict.upper()}</p>
                    <p><strong>Score:</strong> {agent_output.score:.3f}</p>
                    <p><strong>Reasoning:</strong> {agent_output.reasoning}</p>
                </div>
                """

        html += """
            </div>
        </body>
        </html>
        """

        with open(save_path, 'w') as f:
            f.write(html)

        return html


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Explainability Module")
    print("="*80)
    print("\n✅ Features:")
    print("   • Grad-CAM visualization")
    print("   • Attention heatmaps")
    print("   • Saliency maps")
    print("   • Token importance highlighting")
    print("   • HTML report generation")
    print("\n" + "="*80 + "\n")
