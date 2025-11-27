#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug output layer values
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light

def debug_output():
    print("=" * 70)
    print("Debug Output Layer Values")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVNN_Estimator_Light().to(device)
    
    # Create input similar to actual data
    batch_size = 4
    x = torch.randn(batch_size, 2, 100, 100, device=device) * 0.005  # [-0.01, 0.01] range
    
    model.eval()
    
    # Manual forward pass to see intermediate values
    with torch.no_grad():
        # Convert to complex
        z = torch.complex(x[:, 0:1], x[:, 1:2])
        print(f"\n1. Input (complex): shape={z.shape}")
        print(f"   Real range: [{z.real.min():.6f}, {z.real.max():.6f}]")
        print(f"   Imag range: [{z.imag.min():.6f}, {z.imag.max():.6f}]")
        
        # Block 1
        z = model.conv1(z)
        z = model.act1(z)
        z = model.pool1(z)
        print(f"\n2. After Block 1: shape={z.shape}")
        print(f"   Abs range: [{z.abs().min():.6f}, {z.abs().max():.6f}]")
        
        # Block 2
        z = model.conv2(z)
        z = model.act2(z)
        z = model.pool2(z)
        print(f"\n3. After Block 2: shape={z.shape}")
        print(f"   Abs range: [{z.abs().min():.6f}, {z.abs().max():.6f}]")
        
        # Block 3
        z = model.conv3(z)
        z = model.act3(z)
        z = model.pool3(z)
        print(f"\n4. After Block 3: shape={z.shape}")
        print(f"   Abs range: [{z.abs().min():.6f}, {z.abs().max():.6f}]")
        
        # Flatten + FC1
        z = model.flatten(z)
        print(f"\n5. After flatten: shape={z.shape}")
        
        z = model.fc1(z)
        z = model.act_fc1(z)
        print(f"\n6. After FC1: shape={z.shape}")
        print(f"   Abs range: [{z.abs().min():.6f}, {z.abs().max():.6f}]")
        
        # FC out (before sigmoid)
        z_out = model.fc_out(z)
        print(f"\n7. After FC_out (before sigmoid): shape={z_out.shape}")
        print(f"   Complex values: {z_out[0].cpu().numpy()}")
        print(f"   Abs values: {z_out.abs()[0].cpu().numpy()}")
        
        # After to_real (abs)
        abs_out = model.to_real(z_out)
        print(f"\n8. After to_real (abs): shape={abs_out.shape}")
        print(f"   Values: {abs_out[0].cpu().numpy()}")
        
        # After sigmoid
        final_out = torch.sigmoid(abs_out)
        print(f"\n9. After sigmoid (final): shape={final_out.shape}")
        print(f"   Values: {final_out[0].cpu().numpy()}")
        
        print("\n" + "=" * 70)
        print("DIAGNOSIS:")
        print("=" * 70)
        
        if abs_out.min() > 5:
            print(f"PROBLEM: abs values > 5 (min={abs_out.min():.2f})")
            print("  -> sigmoid(x > 5) approx 1.0")
            print("  -> Model output stuck at 1.0!")
            print("\nSOLUTION OPTIONS:")
            print("  1. Use 'real' mode instead of 'abs' in ComplexToReal")
            print("  2. Add a linear layer after to_real to rescale")
            print("  3. Normalize fc_out output before sigmoid")


if __name__ == "__main__":
    debug_output()
