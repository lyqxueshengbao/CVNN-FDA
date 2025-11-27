#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test different output strategies
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light
from complex_layers import ComplexLinear, ComplexToReal

class FixedOutputModel(nn.Module):
    """Model with different output strategy"""
    def __init__(self, base_model, output_mode='real_linear'):
        super().__init__()
        self.base = base_model
        self.output_mode = output_mode
        
        if output_mode == 'real_linear':
            # Replace fc_out with real-valued output
            self.fc_out_real = nn.Linear(256, 2)
            
    def forward(self, x):
        # DataParallel: convert 2-channel real to complex
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Use base model's feature extraction
        x = self.base.conv1(x)
        x = self.base.act1(x)
        x = self.base.pool1(x)
        
        x = self.base.conv2(x)
        x = self.base.act2(x)
        x = self.base.pool2(x)
        
        x = self.base.conv3(x)
        x = self.base.act3(x)
        x = self.base.pool3(x)
        
        x = self.base.flatten(x)
        x = self.base.fc1(x)
        x = self.base.act_fc1(x)
        x = self.base.dropout(x)
        
        if self.output_mode == 'real_linear':
            # Convert complex to real (take abs) before final layer
            x_real = torch.abs(x)  # (batch, 256) real
            out = torch.sigmoid(self.fc_out_real(x_real))
        
        return out


def test_output_strategies():
    print("=" * 70)
    print("Test Different Output Strategies")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    criterion = nn.MSELoss()
    
    # Strategy 1: Original (abs + sigmoid) - Already shown to fail
    print("\n[Strategy 1] Original: ComplexToReal(abs) + sigmoid")
    print("-" * 60)
    model1 = CVNN_Estimator_Light().to(device)
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    
    model1.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        opt1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        loss.backward()
        opt1.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}, Out=[{out.min().item():.3f},{out.max().item():.3f}]")
    
    # Strategy 2: Real linear output
    print("\n[Strategy 2] Real linear output: abs(fc1) -> Linear(real) -> sigmoid")
    print("-" * 60)
    base_model = CVNN_Estimator_Light().to(device)
    model2 = FixedOutputModel(base_model, output_mode='real_linear').to(device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    
    model2.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        opt2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        loss.backward()
        opt2.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}, Out=[{out.min().item():.3f},{out.max().item():.3f}]")
    
    # Strategy 3: Real part only (no abs)
    print("\n[Strategy 3] Real part output: ComplexLinear -> real() -> Linear -> sigmoid")
    print("-" * 60)
    
    class RealPartModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.fc_out_real = nn.Linear(256, 2)
            
        def forward(self, x):
            if not x.is_complex() and x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
            
            x = self.base.conv1(x)
            x = self.base.act1(x)
            x = self.base.pool1(x)
            
            x = self.base.conv2(x)
            x = self.base.act2(x)
            x = self.base.pool2(x)
            
            x = self.base.conv3(x)
            x = self.base.act3(x)
            x = self.base.pool3(x)
            
            x = self.base.flatten(x)
            x = self.base.fc1(x)
            x = self.base.act_fc1(x)
            x = self.base.dropout(x)
            
            # Take real part only
            x_real = x.real
            out = torch.sigmoid(self.fc_out_real(x_real))
            return out
    
    base_model3 = CVNN_Estimator_Light().to(device)
    model3 = RealPartModel(base_model3).to(device)
    opt3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    
    model3.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        opt3.zero_grad()
        out = model3(x)
        loss = criterion(out, y)
        loss.backward()
        opt3.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}, Out=[{out.min().item():.3f},{out.max().item():.3f}]")
    
    # Strategy 4: Concat real+imag
    print("\n[Strategy 4] Concat output: fc1 -> concat(real, imag) -> Linear(512->2) -> sigmoid")
    print("-" * 60)
    
    class ConcatModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.fc_out_real = nn.Linear(512, 2)  # 256*2 = 512
            
        def forward(self, x):
            if not x.is_complex() and x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
            
            x = self.base.conv1(x)
            x = self.base.act1(x)
            x = self.base.pool1(x)
            
            x = self.base.conv2(x)
            x = self.base.act2(x)
            x = self.base.pool2(x)
            
            x = self.base.conv3(x)
            x = self.base.act3(x)
            x = self.base.pool3(x)
            
            x = self.base.flatten(x)
            x = self.base.fc1(x)
            x = self.base.act_fc1(x)
            x = self.base.dropout(x)
            
            # Concat real and imag
            x_concat = torch.cat([x.real, x.imag], dim=-1)
            out = torch.sigmoid(self.fc_out_real(x_concat))
            return out
    
    base_model4 = CVNN_Estimator_Light().to(device)
    model4 = ConcatModel(base_model4).to(device)
    opt4 = torch.optim.Adam(model4.parameters(), lr=1e-3)
    
    model4.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        opt4.zero_grad()
        out = model4(x)
        loss = criterion(out, y)
        loss.backward()
        opt4.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}, Out=[{out.min().item():.3f},{out.max().item():.3f}]")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    test_output_strategies()
