#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ModReLU bias effect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_layers import ComplexConv2d, ComplexLinear, ComplexAvgPool2d, ComplexFlatten, ModReLU

def test_modrelu_effect():
    print("=" * 70)
    print("Test ModReLU Bias Effect")
    print("=" * 70)
    
    # Create a simple complex input
    x = torch.randn(4, 1, 10, 10) * 0.005  # Small values
    z = torch.complex(x, torch.randn_like(x) * 0.005)
    
    print(f"\nInput: |z| range = [{z.abs().min():.6f}, {z.abs().max():.6f}]")
    print(f"       |z| mean = {z.abs().mean():.6f}")
    
    # Test ModReLU with different biases
    for bias in [0.0, 0.1, 0.5, 1.0]:
        modrelu = ModReLU(num_features=1, bias_init=bias)
        out = modrelu(z)
        
        # Check if activation is effective
        # ReLU(|z| + bias) - if |z| + bias > 0, output = |z| + bias
        # With small |z| (~0.005) and positive bias, ReLU always passes
        
        print(f"\nModReLU(bias={bias}):")
        print(f"  Output |z| range = [{out.abs().min():.6f}, {out.abs().max():.6f}]")
        print(f"  Output |z| mean = {out.abs().mean():.6f}")
        
        # Check gradient flow
        z_input = z.clone().requires_grad_(True)
        out = modrelu(z_input)
        loss = out.abs().mean()
        loss.backward()
        print(f"  Input gradient norm = {z_input.grad.abs().mean():.8f}")
    
    print("\n" + "=" * 70)
    print("Analysis: With positive bias, ReLU(|z| + bias) = |z| + bias always")
    print("This means ModReLU doesn't add nonlinearity!")
    print("=" * 70)
    
    # Test with negative bias
    print("\n\nTesting ModReLU with NEGATIVE bias:")
    print("-" * 50)
    
    for bias in [-0.001, -0.01, -0.1]:
        modrelu = ModReLU(num_features=1, bias_init=bias)
        out = modrelu(z)
        
        # With negative bias, some outputs will be zeroed
        num_zeros = (out.abs() < 1e-6).sum().item()
        total = out.numel()
        
        print(f"\nModReLU(bias={bias}):")
        print(f"  Output |z| range = [{out.abs().min():.6f}, {out.abs().max():.6f}]")
        print(f"  Zeros: {num_zeros}/{total} ({100*num_zeros/total:.1f}%)")


if __name__ == "__main__":
    test_modrelu_effect()
