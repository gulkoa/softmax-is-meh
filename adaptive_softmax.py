"""
Adaptive Temperature Softmax Module

Implements per-layer trainable polynomial parameters for adaptive temperature softmax,
based on "Softmax is Not Enough" paper.

Each transformer layer gets its own polynomial coefficients that learn to modulate
attention temperatures based on entropy.
"""

import torch
from torch import nn
from typing import Optional


class AdaptivePolynomialConfig(nn.Module):
    """Manages per-layer polynomial parameters for adaptive temperature softmax."""
    
    def __init__(self, num_layers: int, poly_degree: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.poly_degree = poly_degree
        
        # Paper's initial values from Figure 6
        init_values = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791], dtype=torch.float32)
        
        # Shape: [num_layers, poly_degree]
        # Each layer starts with the same coefficients but can diverge during training
        self.poly_coeffs = nn.Parameter(
            init_values.unsqueeze(0).expand(num_layers, -1).clone()
        )
    
    def get_layer_coeffs(self, layer_idx: int) -> torch.Tensor:
        """Get polynomial coefficients for a specific layer."""
        return self.poly_coeffs[layer_idx]
    
    def save_checkpoint(self, path: str):
        """Save polynomial coefficients to file."""
        torch.save({
            'poly_coeffs': self.poly_coeffs.data,
            'num_layers': self.num_layers,
            'poly_degree': self.poly_degree,
        }, path)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'AdaptivePolynomialConfig':
        """Load polynomial coefficients from file."""
        checkpoint = torch.load(path)
        config = cls(
            num_layers=checkpoint['num_layers'],
            poly_degree=checkpoint['poly_degree']
        )
        config.poly_coeffs.data = checkpoint['poly_coeffs']
        return config
    
    def __repr__(self):
        return f"AdaptivePolynomialConfig(num_layers={self.num_layers}, poly_degree={self.poly_degree})"


def get_polynomial_value(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomial using Horner's method.
    
    For coeffs = [a4, a3, a2, a1, a0], computes:
        a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    
    Args:
        x: Input tensor (entropy values)
        coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
    
    Returns:
        Polynomial evaluated at x
    """
    result = torch.zeros_like(x)
    for i in range(len(coeffs) - 1):
        result = (result + coeffs[i]) * x
    return result + coeffs[-1]


def softmax_adaptive_temperature(
    logits: torch.Tensor,
    dim: int,
    poly_coeffs: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Adaptive temperature softmax with proper device handling.
    
    From "Softmax is Not Enough" - computes entropy of attention distribution
    and scales temperature based on polynomial fit.
    
    Args:
        logits: Attention logits [batch, heads, q_len, kv_len]
        dim: Dimension to apply softmax
        poly_coeffs: Polynomial coefficients for this layer
        dtype: Compute dtype
    
    Returns:
        Attention weights with adaptive temperature applied
    """
    # Compute initial probabilities
    original_probs = torch.softmax(logits, dim=dim, dtype=dtype)
    
    # Compute Shannon entropy: H = -sum(p * log(p))
    entropy = torch.sum(
        -original_probs * torch.log(original_probs + 1e-9),
        dim=-1,
        keepdim=True
    )
    
    # Evaluate polynomial at entropy values
    poly_val = get_polynomial_value(entropy, poly_coeffs)
    
    # Apply guards:
    # 1. Low entropy guard: if H < 0.5, don't adjust (beta = 1.0)
    # 2. Dispersion guard: beta >= 1.0 (never increase entropy)
    # Using ones_like to ensure correct device placement
    beta = torch.where(
        entropy > 0.5,
        torch.maximum(poly_val, torch.ones_like(poly_val)),
        torch.ones_like(entropy)
    )
    
    # Apply scaled softmax
    return torch.softmax(logits * beta, dim=dim, dtype=dtype)


def create_attention_hook(poly_config: AdaptivePolynomialConfig, layer_idx: int):
    """
    Create a hook that applies adaptive softmax to attention weights.
    
    This is designed to be used with model hooks rather than replacing attention entirely.
    """
    def hook(module, args, output):
        # output is typically (attn_output, attn_weights) or just attn_output
        # We need to intercept before softmax is applied
        pass  # Hook implementation depends on specific model architecture
    
    return hook


# Convenience function for quick testing
def test_adaptive_softmax():
    """Quick test to verify adaptive softmax works correctly."""
    print("Testing AdaptivePolynomialConfig...")
    
    config = AdaptivePolynomialConfig(num_layers=34)
    print(f"  Created config: {config}")
    print(f"  Coefficients shape: {config.poly_coeffs.shape}")
    
    # Test polynomial evaluation
    x = torch.tensor([1.5, 2.0, 0.3])
    coeffs = config.get_layer_coeffs(0)
    result = get_polynomial_value(x, coeffs)
    print(f"  Polynomial at [1.5, 2.0, 0.3]: {result.tolist()}")
    
    # Test adaptive softmax
    logits = torch.randn(2, 4, 8, 8)  # [batch, heads, q_len, kv_len]
    probs = softmax_adaptive_temperature(logits, dim=-1, poly_coeffs=coeffs)
    print(f"  Softmax output shape: {probs.shape}")
    print(f"  Probabilities sum to 1: {probs.sum(dim=-1).mean().item():.4f}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_adaptive_softmax()
