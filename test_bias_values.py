#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test different bias values for ModReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_layers import (
    ComplexConv2d, ComplexLinear, ComplexAvgPool2d, 
    ComplexFlatten, ComplexDropout, ModReLU
)

class TestModel(nn.Module):
    """Test model with configurable ModReLU bias"""
    def __init__(self, bias_init=0.5):
        super().__init__()
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = ModReLU(num_features=16, bias_init=bias_init)
        self.pool1 = ComplexAvgPool2d(kernel_size=4, stride=4)
        
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = ModReLU(num_features=32, bias_init=bias_init)
        self.pool2 = ComplexAvgPool2d(kernel_size=5, stride=5)
        
        self.flatten = ComplexFlatten()
        self.fc1 = ComplexLinear(32 * 5 * 5, 64)
        self.act_fc1 = ModReLU(num_features=64, bias_init=bias_init)
        
        self.fc_out = nn.Linear(128, 2)  # concat real+imag
        
    def forward(self, x):
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        
        x_concat = torch.cat([x.real, x.imag], dim=-1)
        out = torch.sigmoid(self.fc_out(x_concat))
        return out


def test_bias_values():
    print("=" * 70)
    print("Test Different ModReLU Bias Values")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    batch_size = 32
    num_steps = 50
    
    bias_values = [0.5, 0.1, 0.0, -0.001, -0.005]
    
    for bias in bias_values:
        print(f"\n[bias_init = {bias}]")
        print("-" * 50)
        
        model = TestModel(bias_init=bias).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        losses = []
        
        for step in range(num_steps):
            x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
            y = torch.rand(batch_size, 2, device=device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 15 == 0:
                print(f"  Step {step+1:3d}: Loss={loss.item():.4f}, "
                      f"Out=[{out.min().item():.3f},{out.max().item():.3f}]")
        
        # Summary
        if losses[-1] < losses[0] * 0.9:
            trend = "IMPROVING"
        elif losses[-1] > losses[0] * 1.1:
            trend = "DEGRADING"
        else:
            trend = "STABLE"
        
        print(f"\n  Summary: {losses[0]:.4f} -> {losses[-1]:.4f} ({trend})")
        
        # Check output range
        model.eval()
        with torch.no_grad():
            x_test = torch.randn(32, 2, 100, 100, device=device) * 0.005
            out_test = model(x_test)
            print(f"  Final output range: [{out_test.min():.3f}, {out_test.max():.3f}]")


if __name__ == "__main__":
    test_bias_values()
