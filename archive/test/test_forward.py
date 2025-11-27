#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug model forward pass, comparing two datasets
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light
from dataset import create_dataloaders
from dataset_cached import create_dataloaders_cached

def test_forward():
    """Test model forward pass"""
    
    print("=" * 60)
    print("Model Forward Pass Test")
    print("=" * 60)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVNN_Estimator_Light(input_size=100).to(device)
    print(f"\nDevice: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create small datasets
    batch_size = 8
    
    print("\n[1] Testing dynamic dataset...")
    train_dyn, _, _ = create_dataloaders(
        train_size=100,
        val_size=10,
        test_size=10,
        batch_size=batch_size,
        num_workers=0
    )
    
    batch_dyn = next(iter(train_dyn))
    R_dyn, labels_dyn, _ = batch_dyn
    R_dyn = R_dyn.to(device)
    labels_dyn = labels_dyn.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output_dyn = model(R_dyn)
    
    print(f"\nDynamic dataset:")
    print(f"  Input shape: {R_dyn.shape}")
    print(f"  Output shape: {output_dyn.shape}")
    print(f"  Output range: [{output_dyn.min():.4f}, {output_dyn.max():.4f}]")
    print(f"  Output mean: {output_dyn.mean():.4f}")
    print(f"  Output sample: {output_dyn[0].cpu().numpy()}")
    print(f"  True label: {labels_dyn[0].cpu().numpy()}")
    
    print("\n[2] Testing cached dataset...")
    train_cached, _, _ = create_dataloaders_cached(
        train_size=100,
        val_size=10,
        test_size=10,
        batch_size=batch_size,
        num_workers=0,
        verbose=False
    )
    
    batch_cached = next(iter(train_cached))
    R_cached, labels_cached, _ = batch_cached
    R_cached = R_cached.to(device)
    labels_cached = labels_cached.to(device)
    
    with torch.no_grad():
        output_cached = model(R_cached)
    
    print(f"\nCached dataset:")
    print(f"  Input shape: {R_cached.shape}")
    print(f"  Output shape: {output_cached.shape}")
    print(f"  Output range: [{output_cached.min():.4f}, {output_cached.max():.4f}]")
    print(f"  Output mean: {output_cached.mean():.4f}")
    print(f"  Output sample: {output_cached[0].cpu().numpy()}")
    print(f"  True label: {labels_cached[0].cpu().numpy()}")
    
    # Compute loss
    criterion = nn.MSELoss()
    loss_dyn = criterion(output_dyn, labels_dyn)
    loss_cached = criterion(output_cached, labels_cached)
    
    print("\n" + "=" * 60)
    print("Loss comparison")
    print("=" * 60)
    print(f"Dynamic dataset MSE Loss: {loss_dyn.item():.6f}")
    print(f"Cached dataset MSE Loss: {loss_cached.item():.6f}")
    
    # Test single training step
    print("\n" + "=" * 60)
    print("Single step training test")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train with dynamic data for a few steps
    print("\nDynamic dataset training:")
    for i in range(3):
        batch = next(iter(train_dyn))
        R, labels, _ = batch
        R = R.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(R)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}: Loss = {loss.item():.6f}")
    
    # Re-initialize model
    model2 = CVNN_Estimator_Light(input_size=100).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    model2.train()
    
    print("\nCached dataset training:")
    for i in range(3):
        batch = next(iter(train_cached))
        R, labels, _ = batch
        R = R.to(device)
        labels = labels.to(device)
        
        optimizer2.zero_grad()
        output = model2(R)
        loss = criterion(output, labels)
        loss.backward()
        optimizer2.step()
        print(f"  Step {i+1}: Loss = {loss.item():.6f}")
    
    # Check gradients
    print("\n" + "=" * 60)
    print("Gradient check")
    print("=" * 60)
    
    # Gradients from last step
    print("\nDynamic dataset model gradient stats:")
    total_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad += grad_norm
            if "conv" in name and "weight" in name:
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
    print(f"  Total grad norm: {total_grad:.6f}")
    
    print("\nCached dataset model gradient stats:")
    total_grad = 0
    for name, param in model2.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad += grad_norm
            if "conv" in name and "weight" in name:
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
    print(f"  Total grad norm: {total_grad:.6f}")


if __name__ == "__main__":
    test_forward()
