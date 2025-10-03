#!/usr/bin/env python3
"""
Custom coarse inference example using your custom trained model.

This script demonstrates how to run inference with your custom cBottle model,
similar to the original coarse_inference.py but using the custom_loader.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt

# Add the current directory to path so we can import custom_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from custom_loader import load_custom_model
from cbottle.visualizations import visualize
from cbottle.datasets.dataset_3d import get_dataset


def run_custom_inference(checkpoint_path: str, output_dir: str = "./outputs"):
    """
    Run inference with your custom model.
    
    Args:
        checkpoint_path: Path to your custom checkpoint file
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading custom model from: {checkpoint_path}")
    
    # Load your custom model
    model = load_custom_model(
        checkpoint_path=checkpoint_path,
        model_name="my-custom-model",
        sigma_min=0.02,
        sigma_max=200.0,
        num_steps=18
    )
    
    print(f"Model loaded successfully!")
    print(f"Available channels: {model.batch_info.channels}")
    
    # Load dataset (same as original example)
    print("Loading dataset...")
    ds = get_dataset(dataset="amip")
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    
    print("Running inference...")
    
    # Run inference - basic sampling
    out, coords = model.sample(batch)
    
    print(f"Generated output shape: {out.shape}")
    print(f"Output coordinates channels: {coords.batch_info.channels}")
    
    # Visualize different channels
    visualize_channels(out, coords, output_dir)
    
    # If your model supports guidance (like hurricane guidance), you can also try:
    # run_guidance_inference(model, batch, output_dir)
    
    return out, coords


def visualize_channels(out, coords, output_dir):
    """Visualize different channels from the generated output."""
    channels = coords.batch_info.channels
    
    print(f"Visualizing {len(channels)} channels...")
    
    for i, channel in enumerate(channels):
        try:
            plt.figure(figsize=(10, 8))
            visualize(out[0, i, 0], nest=True)
            plt.title(f"Generated {channel}")
            plt.colorbar()
            
            output_path = os.path.join(output_dir, f"generated_{channel}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved {channel} visualization to {output_path}")
            
        except Exception as e:
            print(f"  ❌ Failed to visualize {channel}: {e}")


def run_guidance_inference(model, batch, output_dir):
    """Run inference with guidance (if supported by your model)."""
    print("Trying guidance inference...")
    
    try:
        # Example: Hurricane guidance at specific coordinates
        # 27.6648° N, 81.5158° W (same as original example)
        guidance_pixels = model.get_guidance_pixels([-81.5], [27.6])
        
        out_guided, coords = model.sample(
            batch, 
            guidance_pixels=guidance_pixels
        )
        
        # Visualize guided output
        if "uas" in coords.batch_info.channels:
            c = coords.batch_info.channels.index("uas")
            plt.figure(figsize=(10, 8))
            visualize(out_guided[0, c, 0], nest=True)
            plt.title("Hurricane Guidance - UAS")
            plt.colorbar()
            
            output_path = os.path.join(output_dir, "hurricane_guidance.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved hurricane guidance visualization to {output_path}")
        
    except Exception as e:
        print(f"❌ Guidance inference failed (this is normal if your model doesn't support guidance): {e}")


def run_multiple_samples(checkpoint_path: str, num_samples: int = 3, output_dir: str = "./outputs"):
    """Run multiple samples to see variability in generation."""
    print(f"Running {num_samples} samples...")
    
    # Load model
    model = load_custom_model(checkpoint_path=checkpoint_path)
    
    # Load dataset
    ds = get_dataset(dataset="amip")
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        
        # Run inference
        out, coords = model.sample(batch)
        
        # Save a few key channels
        key_channels = ["uas", "vas", "tas", "psl"]  # Adjust based on your model's channels
        available_channels = coords.batch_info.channels
        
        for channel in key_channels:
            if channel in available_channels:
                c = available_channels.index(channel)
                
                plt.figure(figsize=(10, 8))
                visualize(out[0, c, 0], nest=True)
                plt.title(f"Sample {i+1} - {channel}")
                plt.colorbar()
                
                output_path = os.path.join(output_dir, f"sample_{i+1}_{channel}.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ Saved sample {i+1} {channel} to {output_path}")


if __name__ == "__main__":
    # Update this path to your actual checkpoint file
    checkpoint_path = "/path/to/your/custom-model.checkpoint"
    
    # Update this to your desired output directory
    output_dir = "./custom_model_outputs"
    
    print("Custom cBottle Inference Example")
    print("=" * 40)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
        sys.exit(1)
    
    try:
        # Run basic inference
        print("\n1. Running basic inference...")
        out, coords = run_custom_inference(checkpoint_path, output_dir)
        
        # Run multiple samples
        print("\n2. Running multiple samples...")
        run_multiple_samples(checkpoint_path, num_samples=3, output_dir=output_dir)
        
        print(f"\n✅ All done! Check the outputs in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
