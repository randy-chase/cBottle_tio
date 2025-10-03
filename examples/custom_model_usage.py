#!/usr/bin/env python3
"""
Example usage of custom model loader.

This script demonstrates how to load and use custom trained cBottle models.
"""

import os
import torch
from datetime import datetime
from custom_loader import load_custom_model, load_custom_moe_model, inspect_checkpoint


def example_single_model():
    """Example of loading a single custom model."""
    print("=== Loading Single Custom Model ===")
    
    # Replace with your actual checkpoint path
    checkpoint_path = "/path/to/your/custom-model.checkpoint"
    
    # First, inspect the checkpoint to see what it contains
    if os.path.exists(checkpoint_path):
        print("Inspecting checkpoint...")
        info = inspect_checkpoint(checkpoint_path)
        print("Available channels:", info["batch_info"]["channels"])
        print("Model architecture:", info["model_config"]["architecture"])
        print("Model channels:", info["model_config"]["model_channels"])
    
    # Load the model
    try:
        model = load_custom_model(
            checkpoint_path=checkpoint_path,
            model_name="my-custom-model",
            sigma_min=0.02,
            sigma_max=200.0,
            num_steps=18
        )
        
        print("Model loaded successfully!")
        print("Available variables:", model.batch_info.channels)
        
        # Example usage (you'll need to provide appropriate batch data)
        # batch = create_your_batch()  # Create your batch data
        # output, coords = model.sample(batch)
        
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
    except AttributeError as e:
        print(f"Model loading error: {e}")
        print("This usually means the checkpoint doesn't have the required batch_info.")
    except Exception as e:
        print(f"Unexpected error loading model: {e}")


def example_moe_model():
    """Example of loading a Mixture of Experts model."""
    print("\n=== Loading MoE Custom Model ===")
    
    # Replace with your actual checkpoint paths
    checkpoint_paths = [
        "/path/to/expert1.checkpoint",
        "/path/to/expert2.checkpoint", 
        "/path/to/expert3.checkpoint"
    ]
    sigma_thresholds = [100.0, 10.0]  # One less than number of experts
    
    # Check if files exist
    existing_paths = [p for p in checkpoint_paths if os.path.exists(p)]
    if not existing_paths:
        print("No checkpoint files found. Please update the paths.")
        return
    
    try:
        model = load_custom_moe_model(
            checkpoint_paths=existing_paths,
            sigma_thresholds=sigma_thresholds[:len(existing_paths)-1],
            model_name="my-custom-moe-model"
        )
        
        print("MoE model loaded successfully!")
        print("Available variables:", model.batch_info.channels)
        
    except AttributeError as e:
        print(f"MoE model loading error: {e}")
        print("This usually means the checkpoint doesn't have the required batch_info.")
    except Exception as e:
        print(f"Error loading MoE model: {e}")


def example_with_classifier():
    """Example of loading a model with a separate classifier for guidance."""
    print("\n=== Loading Model with Classifier ===")
    
    checkpoint_path = "/path/to/your/model.checkpoint"
    classifier_path = "/path/to/your/classifier.checkpoint"
    
    if not os.path.exists(checkpoint_path):
        print("Model checkpoint not found. Please update the path.")
        return
    
    try:
        model = load_custom_model(
            checkpoint_path=checkpoint_path,
            separate_classifier_path=classifier_path if os.path.exists(classifier_path) else None,
            model_name="model-with-classifier"
        )
        
        print("Model with classifier loaded successfully!")
        print("Available variables:", model.batch_info.channels)
        
        # Example of using guidance (if classifier is available)
        if model.separate_classifier is not None:
            print("Classifier available for guidance!")
            # You can now use guidance_pixels in model.sample()
        else:
            print("No classifier available.")
            
    except AttributeError as e:
        print(f"Model with classifier loading error: {e}")
        print("This usually means the checkpoint doesn't have the required batch_info.")
    except Exception as e:
        print(f"Error loading model with classifier: {e}")


def example_integration_with_existing():
    """Example of integrating with existing cBottle workflow."""
    print("\n=== Integration Example ===")
    
    # This shows how you might integrate your custom model
    # with the existing cBottle inference patterns
    
    try:
        # Load your custom model
        model = load_custom_model("/path/to/your/model.checkpoint")
        
        # Get available variables
        available_vars = model.batch_info.channels
        print(f"Your model supports {len(available_vars)} variables:")
        for i, var in enumerate(available_vars):
            print(f"  {i}: {var}")
        
        # Example of creating a batch (you'll need to adapt this to your data)
        # This is just a placeholder - you'll need real atmospheric data
        batch_size = 1
        time_length = model.time_length
        num_channels = len(available_vars)
        num_pixels = model.net.domain.numel()
        
        # Create dummy batch (replace with your actual data loading)
        dummy_batch = {
            "target": torch.randn(batch_size, num_channels, time_length, num_pixels),
            "labels": torch.zeros(batch_size, 2),  # Assuming 2 dataset types
            "condition": torch.randn(batch_size, 0, time_length, num_pixels),  # No condition channels
            "second_of_day": torch.randint(0, 86400, (batch_size,)),
            "day_of_year": torch.randint(1, 366, (batch_size,))
        }
        
        print("Batch created successfully!")
        print("You can now use model.sample(dummy_batch) to generate data")
        
    except AttributeError as e:
        print(f"Integration example error: {e}")
        print("This usually means the checkpoint doesn't have the required batch_info.")
    except Exception as e:
        print(f"Error in integration example: {e}")


if __name__ == "__main__":
    print("Custom Model Loader Examples")
    print("=" * 40)
    
    # Run examples
    example_single_model()
    example_moe_model() 
    example_with_classifier()
    example_integration_with_existing()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nTo use with your own models:")
    print("1. Update the checkpoint paths in the examples")
    print("2. Ensure your checkpoints are in the correct format")
    print("3. Adapt the batch creation to your specific data")
    print("4. Use model.sample() or model.infill() as needed")
