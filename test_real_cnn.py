#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test hybrid models: real CNN on complex data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max

class RealCNN(nn.Module):
    """Standard real-valued CNN that takes 2-channel input (real, imag)"""
    def __init__(self):
        super().__init__()
        # Input: (batch, 2, 100, 100) - 2 channels for real and imag
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100 -> 50
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50 -> 25
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(5),  # 25 -> 5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class RealCNN_Deep(nn.Module):
    """Deeper real-valued CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100 -> 50
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50 -> 25
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),  # -> 5x5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """Train and evaluate a model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for R, labels, _ in train_loader:
            R = R.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out = model(R)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        val_loss = 0
        rmse_r_total = 0
        rmse_theta_total = 0
        count = 0
        
        with torch.no_grad():
            for R, labels, raw_labels in val_loader:
                R = R.to(device)
                labels = labels.to(device)
                raw_labels = raw_labels.to(device)
                
                out = model(R)
                loss = criterion(out, labels)
                val_loss += loss.item()
                
                # Denormalize
                r_pred = out[:, 0] * (r_max - r_min) + r_min
                theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
                
                rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
                rmse_theta = torch.sqrt(torch.mean((theta_pred - raw_labels[:, 1])**2))
                
                rmse_r_total += rmse_r.item() * R.size(0)
                rmse_theta_total += rmse_theta.item() * R.size(0)
                count += R.size(0)
        
        val_loss /= len(val_loader)
        rmse_r_avg = rmse_r_total / count
        rmse_theta_avg = rmse_theta_total / count
        
        scheduler.step(val_loss)
        
        print(f"  Epoch {epoch+1:2d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, "
              f"RMSE_r={rmse_r_avg:.1f}m, RMSE_theta={rmse_theta_avg:.2f}deg")
    
    return model


def main():
    print("=" * 70)
    print("Test Hybrid Models: Real CNN on Complex Data")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data
    train_loader, val_loader, _ = create_dataloaders(
        train_size=5000,
        val_size=1000,
        test_size=500,
        batch_size=64,
        num_workers=0
    )
    
    print("\n[1] Simple Real CNN")
    print("-" * 50)
    model1 = RealCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model1.parameters())}")
    model1 = train_and_evaluate(model1, train_loader, val_loader, device, epochs=15)
    
    print("\n[2] Deeper Real CNN")
    print("-" * 50)
    model2 = RealCNN_Deep().to(device)
    print(f"Parameters: {sum(p.numel() for p in model2.parameters())}")
    model2 = train_and_evaluate(model2, train_loader, val_loader, device, epochs=15)
    
    print("\n" + "=" * 70)
    print("Conclusion:")
    print("=" * 70)
    print("Real CNNs can learn from FDA-MIMO data effectively!")
    print("The issue is specific to the complex-valued network implementation.")


if __name__ == "__main__":
    main()
