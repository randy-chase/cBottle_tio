#!/usr/bin/env python3
"""
Example of using regression guidance with the proper cBottle package imports.

This script demonstrates how to use the regression guidance system
after installing cBottle_tio as a package.
"""

import torch
import cbottle


def main():
    """
    Main example function demonstrating regression guidance with proper package imports.
    """
    print("Regression Guidance with cBottle Package")
    print("=" * 45)
    
    # Configuration
    checkpoint_path = "/path/to/your/custom-model.checkpoint"  # Update this path
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
        print("\nFor demonstration, we'll show the setup without loading the model:")
        
        # Show how to create synthetic observations
        print("\nCreating synthetic observations for demonstration...")
        observation_data, observation_locations = cbottle.create_synthetic_observations(
            observation_variables=['T850', 'T500', 'T300'],
            num_obs=50
        )
        
        print(f"✅ Synthetic observations created:")
        print(f"  - Shape: {observation_data.shape}")
        print(f"  - Variables: ['T850', 'T500', 'T300']")
        print(f"  - Temperature range: {observation_data.min():.1f}K - {observation_data.max():.1f}K")
        
        return
    
    try:
        # Quick setup with AMIP dataset using the package
        print("Setting up regression guidance with AMIP dataset...")
        model, batch = cbottle.quick_regression_guidance_setup(
            checkpoint_path=checkpoint_path,
            observation_variables=['T850', 'T500', 'T300'],
            num_obs=100,
            guidance_scale=1.0,
            use_amip=True  # Uses real AMIP data with proper time encoding and SST
        )
        
        print("✅ Setup complete!")
        print(f"Model loaded: {type(model).__name__}")
        print(f"Batch shape: {batch['target'].shape}")
        
        # Perform data assimilation
        print("Performing data assimilation...")
        output, coords = model.sample(batch, seed=42)
        
        print(f"✅ Data assimilation complete!")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_package_imports():
    """
    Test that all regression guidance modules can be imported from the package.
    """
    print("Testing package imports...")
    
    try:
        # Test importing from the main package
        from cbottle import (
            RegressionGuidedCBottle3d,
            load_custom_model_with_regression_guidance,
            RegressionGuidance,
            quick_regression_guidance_setup,
            setup_regression_guidance_with_amip
        )
        print("✅ All regression guidance modules imported successfully from cbottle package")
        
        # Test creating a RegressionGuidance instance
        observation_data = torch.randn(50, 3) * 10 + 273.15
        observation_locations = torch.randint(0, 1000, (50,))
        observation_variables = ['T850', 'T500', 'T300']
        
        guidance = RegressionGuidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=0.1,
            guidance_scale=1.0,
        )
        print("✅ RegressionGuidance instance created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import from cbottle package: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing package imports: {e}")
        return False


def demonstrate_usage():
    """
    Demonstrate the proper usage pattern.
    """
    print("\nProper Usage Pattern:")
    print("-" * 25)
    print("""
# After installing cBottle_tio as a package, you can use:

import cbottle

# Quick setup with AMIP dataset
model, batch = cbottle.quick_regression_guidance_setup(
    checkpoint_path="/path/to/your/model.checkpoint",
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100,
    guidance_scale=1.0,
    use_amip=True
)

# Perform data assimilation
output, coords = model.sample(batch, seed=42)
""")
    
    print("Or for more control:")
    print("""
# Manual setup
from cbottle import load_custom_model_with_regression_guidance, setup_regression_guidance_with_amip
from cbottle.datasets.dataset_3d import get_dataset

# Load model
model = load_custom_model_with_regression_guidance(
    checkpoint_path="/path/to/your/model.checkpoint"
)

# Load AMIP dataset
ds = get_dataset(dataset="amip")
loader = torch.utils.data.DataLoader(ds, batch_size=1)
batch = next(iter(loader))

# Set up regression guidance
setup_regression_guidance_with_amip(
    model=model,
    amip_batch=batch,
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100
)

# Sample
output, coords = model.sample(batch)
""")


if __name__ == "__main__":
    import os
    
    print("Regression Guidance Package Test")
    print("=" * 35)
    
    if test_package_imports():
        print("\n" + "=" * 35)
        main()
    else:
        print("\n❌ Package import test failed.")
        print("Make sure cBottle_tio is properly installed as a package.")
    
    demonstrate_usage()
