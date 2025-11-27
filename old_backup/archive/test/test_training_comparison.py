#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete training comparison between dynamic and cached datasets
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light
from dataset import create_dataloaders
from dataset_cached import create_dataloaders_cached
from config import r_min, r_max, theta_min, theta_max
import time

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    for batch_idx, (R, labels, raw_labels) in enumerate(loader):
        R = R.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(R)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_rmse_r = 0
    total_rmse_theta = 0
    n_samples = 0
    
    with torch.no_grad():
        for R, labels, raw_labels in loader:
            R = R.to(device)
            labels = labels.to(device)
            raw_labels = raw_labels.to(device)
            
            output = model(R)
            loss = criterion(output, labels)
            total_loss += loss.item()
            
            # Denormalize predictions
            r_pred = output[:, 0] * (r_max - r_min) + r_min
            theta_pred = output[:, 1] * (theta_max - theta_min) + theta_min
            
            # RMSE
            rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
            rmse_theta = torch.sqrt(torch.mean((theta_pred - raw_labels[:, 1])**2))
            
            total_rmse_r += rmse_r.item() * R.size(0)
            total_rmse_theta += rmse_theta.item() * R.size(0)
            n_samples += R.size(0)
    
    return total_loss / len(loader), total_rmse_r / n_samples, total_rmse_theta / n_samples

def main():
    print("=" * 70)
    print("Full Training Comparison: Dynamic vs Cached Dataset")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    train_size = 2000
    val_size = 500
    batch_size = 64
    epochs = 5
    lr = 1e-3
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
    
    # ===============================================
    # Test Dynamic Dataset
    # ===============================================
    print("\n" + "=" * 70)
    print("Testing DYNAMIC Dataset")
    print("=" * 70)
    
    train_dyn, val_dyn, _ = create_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=100,
        batch_size=batch_size,
        num_workers=0
    )
    
    model_dyn = CVNN_Estimator_Light().to(device)
    criterion = nn.MSELoss()
    optimizer_dyn = torch.optim.Adam(model_dyn.parameters(), lr=lr)
    
    print(f"\nTraining with dynamic dataset:")
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model_dyn, train_dyn, criterion, optimizer_dyn, device)
        val_loss, val_rmse_r, val_rmse_theta = evaluate(model_dyn, val_dyn, criterion, device)
        print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, RMSE_r={val_rmse_r:.1f}m, RMSE_theta={val_rmse_theta:.2f}deg")
    time_dyn = time.time() - start
    print(f"  Total time: {time_dyn:.2f}s")
    
    # ===============================================
    # Test Cached Dataset
    # ===============================================
    print("\n" + "=" * 70)
    print("Testing CACHED Dataset")
    print("=" * 70)
    
    train_cached, val_cached, _ = create_dataloaders_cached(
        train_size=train_size,
        val_size=val_size,
        test_size=100,
        batch_size=batch_size,
        num_workers=0,
        verbose=True
    )
    
    model_cached = CVNN_Estimator_Light().to(device)
    optimizer_cached = torch.optim.Adam(model_cached.parameters(), lr=lr)
    
    print(f"\nTraining with cached dataset:")
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model_cached, train_cached, criterion, optimizer_cached, device)
        val_loss, val_rmse_r, val_rmse_theta = evaluate(model_cached, val_cached, criterion, device)
        print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, RMSE_r={val_rmse_r:.1f}m, RMSE_theta={val_rmse_theta:.2f}deg")
    time_cached = time.time() - start
    print(f"  Total time: {time_cached:.2f}s")
    
    # ===============================================
    # Summary
    # ===============================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dynamic: Time={time_dyn:.2f}s")
    print(f"Cached:  Time={time_cached:.2f}s (speedup: {time_dyn/time_cached:.2f}x)")
    

if __name__ == "__main__":
    main()
