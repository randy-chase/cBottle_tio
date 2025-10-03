#!/usr/bin/env python3
"""
Simple custom inference example - direct adaptation of coarse_inference.py.

This is a minimal example that closely follows the original coarse_inference.py
but uses your custom model instead of the built-in ones.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt

# Add the parent directory to path so we can import custom_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from custom_loader import load_custom_model
from cbottle.visualizations import visualize
from cbottle.datasets.dataset_3d import get_dataset


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Path to your custom checkpoint file
CHECKPOINT_PATH = "/path/to/your/custom-model.checkpoint"

# Output directory for generated images
OUTPUT_DIR = "./outputs"


# ============================================================================
# MAIN INFERENCE CODE
# ============================================================================

def main():
    """Main inference function - adapted from coarse_inference.py"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset (same as original)
    print("Loading dataset...")
    ds = get_dataset(dataset="amip")
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    
    # Load your custom model (replaces cbottle.inference.load())
    print(f"Loading custom model from: {CHECKPOINT_PATH}")
    model = load_custom_model(
        checkpoint_path=CHECKPOINT_PATH,
        model_name="my-custom-model"
    )
    
    print(f"Model loaded! Available channels: {model.batch_info.channels}")
    
    # Basic inference (replaces the original model.sample() calls)
    print("Running inference...")
    out, coords = model.sample(batch)
    
    print(f"Generated output shape: {out.shape}")
    
    # Visualize specific channels (adapt based on your model's channels)
    visualize_output(out, coords)
    
    # Try guidance if supported (optional)
    try_guidance_inference(model, batch)


def visualize_output(out, coords):
    """Visualize the generated output - adapt channel names to your model."""
    
    # Try to visualize common atmospheric variables
    # Adjust these channel names based on what your model actually outputs
    channels_to_visualize = [
        "uas",    # Eastward wind
        "vas",    # Northward wind  
        "tas",    # Temperature
        "psl",    # Sea level pressure
        "rsut",   # Upward shortwave radiation
        "rlut"    # Upward longwave radiation
    ]
    
    available_channels = coords.batch_info.channels
    print(f"Available channels: {available_channels}")
    
    for channel in channels_to_visualize:
        if channel in available_channels:
            c = available_channels.index(channel)
            
            plt.figure(figsize=(12, 8))
            visualize(out[0, c, 0], nest=True)
            plt.title(f"Generated {channel}")
            plt.colorbar()
            
            output_path = os.path.join(OUTPUT_DIR, f"generated_{channel}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved {channel} visualization to {output_path}")
        else:
            print(f"⚠️  Channel {channel} not available in model output")


def try_guidance_inference(model, batch):
    """Try guidance inference if supported by your model."""
    
    print("\nTrying guidance inference...")
    
    try:
        # Hurricane guidance example (same coordinates as original)
        # 27.6648° N, 81.5158° W
        guidance_pixels = model.get_guidance_pixels([-81.5], [27.6])
        
        out_guided, coords = model.sample(
            batch, 
            guidance_pixels=guidance_pixels
        )
        
        # Visualize guided output
        if "uas" in coords.batch_info.channels:
            c = coords.batch_info.channels.index("uas")
            
            plt.figure(figsize=(12, 8))
            visualize(out_guided[0, c, 0], nest=True)
            plt.title("Hurricane Guidance - UAS")
            plt.colorbar()
            
            output_path = os.path.join(OUTPUT_DIR, "hurricane_guidance.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved hurricane guidance to {output_path}")
        else:
            print("⚠️  UAS channel not available for guidance visualization")
            
    except Exception as e:
        print(f"⚠️  Guidance inference not supported or failed: {e}")
        print("   This is normal if your model doesn't support guidance")


if __name__ == "__main__":
    print("Custom cBottle Inference")
    print("=" * 30)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please update the CHECKPOINT_PATH variable with your actual model path.")
        sys.exit(1)
    
    try:
        main()
        print(f"\n✅ Inference completed! Check outputs in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
