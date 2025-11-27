#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick validation of RealCNN_Estimator
"""

import torch
import torch.nn as nn
from model import RealCNN_Estimator
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max
import time

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for R, labels, _ in loader:
        R, labels = R.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(R)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_rmse_r, total_rmse_theta = 0, 0
    n_samples = 0
    
    with torch.no_grad():
        for R, labels, raw_labels in loader:
            R, labels = R.to(device), labels.to(device)
            raw_labels = raw_labels.to(device)
            
            out = model(R)
            loss = criterion(out, labels)
            total_loss += loss.item()
            
            # Denormalize
            r_pred = out[:, 0] * (r_max - r_min) + r_min
            theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
            
            rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
            rmse_theta = torch.sqrt(torch.mean((theta_pred - raw_labels[:, 1])**2))
            
            total_rmse_r += rmse_r.item() * R.size(0)
            total_rmse_theta += rmse_theta.item() * R.size(0)
            n_samples += R.size(0)
    
    return total_loss / len(loader), total_rmse_r / n_samples, total_rmse_theta / n_samples

def main():
    print("=" * 70)
    print("RealCNN_Estimator Validation")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create data
    train_size, val_size = 5000, 1000
    batch_size = 64
    epochs = 10
    
    print(f"\nCreating datasets (train={train_size}, val={val_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=500,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Create model
    model = RealCNN_Estimator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)
    
    best_rmse_r = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse_r, val_rmse_theta = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        
        if val_rmse_r < best_rmse_r:
            best_rmse_r = val_rmse_r
            marker = " *"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"RMSE_r={val_rmse_r:6.1f}m, RMSE_theta={val_rmse_theta:5.2f}deg, LR={lr:.6f}{marker}")
    
    total_time = time.time() - start_time
    
    # Final test evaluation
    test_loss, test_rmse_r, test_rmse_theta = evaluate(model, test_loader, criterion, device)
    
    print("-" * 70)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best validation RMSE_r: {best_rmse_r:.1f}m")
    print(f"Test results: RMSE_r={test_rmse_r:.1f}m, RMSE_theta={test_rmse_theta:.2f}deg")
    
    # Compare with target
    print("\n" + "=" * 70)
    print("Performance vs Target")
    print("=" * 70)
    print(f"Target RMSE_r @ 10dB:  < 5m (excellent < 2m)")
    print(f"Achieved RMSE_r:       {test_rmse_r:.1f}m")
    print(f"Target RMSE_theta:     < 0.5deg (excellent < 0.1deg)")  
    print(f"Achieved RMSE_theta:   {test_rmse_theta:.2f}deg")
    
    if test_rmse_r < 50:
        print("\n>>> Good progress! Model is learning the pattern.")
    elif test_rmse_r < 200:
        print("\n>>> Model is learning but needs more training/tuning.")
    else:
        print("\n>>> Model performance needs improvement.")


if __name__ == "__main__":
    main()
