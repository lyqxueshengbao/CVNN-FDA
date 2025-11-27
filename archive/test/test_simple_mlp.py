#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if the problem is in the data or the model
"""

import torch
import torch.nn as nn
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max

def test_simple_mlp():
    """Test if a simple MLP can learn from the data"""
    print("=" * 70)
    print("Test Simple MLP on Real Data")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data
    train_loader, val_loader, _ = create_dataloaders(
        train_size=2000,
        val_size=500,
        test_size=100,
        batch_size=64,
        num_workers=0
    )
    
    # Simple MLP that takes flattened input
    class SimpleMLP(nn.Module):
        def __init__(self, input_size=2*100*100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = SimpleMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print("\nTraining simple MLP:")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for R, labels, raw_labels in train_loader:
            R = R.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out = model(R)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        val_loss = 0
        val_rmse_r = 0
        val_count = 0
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
                rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
                val_rmse_r += rmse_r.item() * R.size(0)
                val_count += R.size(0)
        
        val_loss /= len(val_loader)
        val_rmse_r /= val_count
        
        print(f"  Epoch {epoch+1:2d}: Train={avg_loss:.4f}, Val={val_loss:.4f}, RMSE_r={val_rmse_r:.1f}m")
        model.train()
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print(f"If RMSE_r stays around {(r_max-r_min)/3.46:.0f}m, the data has no learnable pattern")
    print(f"Random guess RMSE for uniform [0, {r_max}] is approx {(r_max-r_min)/3.46:.0f}m")


if __name__ == "__main__":
    test_simple_mlp()
