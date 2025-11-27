#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why model doesn't learn
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light
import numpy as np

def test_simple_pattern():
    """Test if model can learn a simple pattern"""
    print("=" * 70)
    print("Simple Pattern Learning Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVNN_Estimator_Light().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create simple synthetic data with clear patterns
    batch_size = 32
    
    print("\nTest 1: Constant input -> Constant output")
    print("-" * 50)
    
    # All zeros input should give same output
    x_const = torch.zeros(batch_size, 2, 100, 100, device=device)
    y_const = torch.ones(batch_size, 2, device=device) * 0.5
    
    model.train()
    for i in range(10):
        optimizer.zero_grad()
        out = model(x_const)
        loss = criterion(out, y_const)
        loss.backward()
        optimizer.step()
        if i % 3 == 0:
            print(f"  Step {i+1}: Loss = {loss.item():.6f}, Output mean = {out.mean().item():.4f}")
    
    print("\nTest 2: Random input -> Random output (should NOT learn)")
    print("-" * 50)
    
    model2 = CVNN_Estimator_Light().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    
    model2.train()
    losses = []
    for i in range(10):
        # Random input/output - no pattern to learn
        x_rand = torch.randn(batch_size, 2, 100, 100, device=device) * 0.01
        y_rand = torch.rand(batch_size, 2, device=device)
        
        optimizer2.zero_grad()
        out = model2(x_rand)
        loss = criterion(out, y_rand)
        loss.backward()
        optimizer2.step()
        losses.append(loss.item())
        if i % 3 == 0:
            print(f"  Step {i+1}: Loss = {loss.item():.6f}, Output range = [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    print("\nTest 3: Correlated input-output (should learn)")
    print("-" * 50)
    
    model3 = CVNN_Estimator_Light().to(device)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    
    model3.train()
    losses = []
    for i in range(30):
        # Create correlated data: y depends on mean of x
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.01
        # y is based on sum of diagonal elements (simple pattern)
        y_val = x[:, 0, :10, :10].mean(dim=[1,2]).unsqueeze(1)  # shape (batch, 1)
        y = torch.cat([y_val * 10 + 0.5, y_val * 5 + 0.5], dim=1)  # shape (batch, 2)
        y = torch.clamp(y, 0, 1)
        
        optimizer3.zero_grad()
        out = model3(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer3.step()
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"  Step {i+1}: Loss = {loss.item():.6f}")
    
    print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if losses[-1] < losses[0] * 0.9:
        print("  SUCCESS: Model can learn correlated patterns!")
    else:
        print("  FAILURE: Model cannot learn even simple patterns")
    
    print("\nTest 4: Check gradients with actual data range [-0.01, 0.01]")
    print("-" * 50)
    
    model4 = CVNN_Estimator_Light().to(device)
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=1e-3)
    
    # This matches our actual data range!
    x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005  # [-0.01, 0.01] range
    y = torch.rand(batch_size, 2, device=device)
    
    model4.train()
    optimizer4.zero_grad()
    out = model4(x)
    loss = criterion(out, y)
    loss.backward()
    
    # Check gradients at first conv layer
    grad_conv1_real = model4.conv1.conv_real.weight.grad
    grad_conv1_imag = model4.conv1.conv_imag.weight.grad
    
    print(f"  Input range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  conv1 real grad norm: {grad_conv1_real.norm().item():.8f}")
    print(f"  conv1 imag grad norm: {grad_conv1_imag.norm().item():.8f}")
    
    if grad_conv1_real.norm().item() < 1e-6:
        print("\n  WARNING: Gradients are EXTREMELY SMALL!")
        print("  This explains why the model doesn't learn.")
        print("  The input values are too small (~0.01) causing vanishing gradients.")


if __name__ == "__main__":
    test_simple_pattern()
