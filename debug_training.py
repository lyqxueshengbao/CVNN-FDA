#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug training dynamics
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light

def debug_training():
    print("=" * 70)
    print("Debug Training Dynamics")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVNN_Estimator_Light().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 32
    
    print("\nTraining with random data in actual range [-0.01, 0.01]:")
    print("-" * 60)
    
    model.train()
    for step in range(20):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Check fc_out weights and gradients
        fc_out_weight_norm = model.fc_out.linear_real.weight.norm().item()
        fc_out_grad_norm = model.fc_out.linear_real.weight.grad.norm().item() if model.fc_out.linear_real.weight.grad is not None else 0
        
        # Check conv1 gradients
        conv1_grad_norm = model.conv1.conv_real.weight.grad.norm().item() if model.conv1.conv_real.weight.grad is not None else 0
        
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step+1:3d}: Loss={loss.item():.4f}, "
                  f"Out=[{out.min().item():.3f},{out.max().item():.3f}], "
                  f"fc_out_w={fc_out_weight_norm:.4f}, "
                  f"fc_out_g={fc_out_grad_norm:.6f}, "
                  f"conv1_g={conv1_grad_norm:.8f}")
    
    print("\n" + "=" * 70)
    print("Check: Does changing lr help?")
    print("=" * 70)
    
    model2 = CVNN_Estimator_Light().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)  # Much higher LR
    
    print("\nWith lr=0.1:")
    model2.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005
        y = torch.rand(batch_size, 2, device=device)
        
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer2.step()
        
        if step % 3 == 0:
            print(f"Step {step+1:3d}: Loss={loss.item():.4f}, "
                  f"Out=[{out.min().item():.3f},{out.max().item():.3f}]")
    
    print("\n" + "=" * 70)
    print("Check: Does input scaling help?")
    print("=" * 70)
    
    model3 = CVNN_Estimator_Light().to(device)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    
    print("\nWith scaled input (x * 100 to get [-1, 1] range):")
    model3.train()
    for step in range(10):
        x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.5  # Scaled to [-1, 1]
        y = torch.rand(batch_size, 2, device=device)
        
        optimizer3.zero_grad()
        out = model3(x)
        loss = criterion(out, y)
        loss.backward()
        
        conv1_grad_norm = model3.conv1.conv_real.weight.grad.norm().item()
        
        optimizer3.step()
        
        if step % 3 == 0:
            print(f"Step {step+1:3d}: Loss={loss.item():.4f}, "
                  f"Out=[{out.min().item():.3f},{out.max().item():.3f}], "
                  f"conv1_g={conv1_grad_norm:.6f}")


if __name__ == "__main__":
    debug_training()
