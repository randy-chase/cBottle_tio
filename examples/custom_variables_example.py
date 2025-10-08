#!/usr/bin/env python3
"""
Example of using regression guidance with custom variable configuration.

This script demonstrates how to handle the channel mismatch issue by
providing the correct variable list for your custom model.
"""

import cbottle

def main():
    """
    Example showing how to use custom variables to fix channel mismatch.
    """
    print("Custom Variables Example for Regression Guidance")
    print("=" * 50)
    
    # Your custom model's variables (from your earlier message)
    custom_variables = [
        'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
        'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10',
        'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
        'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
        'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
        'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut',
        'pres_msl', 'pr', 'rsds', 'sst', 'sic'
    ]
    
    print(f"Custom model has {len(custom_variables)} variables")
    print(f"Variables: {custom_variables[:10]}... (showing first 10)")
    
    # Configuration
    checkpoint_path = "/path/to/your/custom-model.checkpoint"  # Update this path
    
    # Check if checkpoint exists
    import os
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
        print("\nFor demonstration, we'll show the setup without loading the model:")
        
        # Show how the setup would work
        print("\nExample usage:")
        print("```python")
        print("import cbottle")
        print("")
        print("# Your custom model's variables")
        print("custom_variables = [")
        print("    'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',")
        print("    'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10',")
        print("    'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',")
        print("    'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',")
        print("    'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',")
        print("    'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut',")
        print("    'pres_msl', 'pr', 'rsds', 'sst', 'sic'")
        print("]")
        print("")
        print("# Quick setup with custom variables")
        print("model, batch = cbottle.quick_regression_guidance_setup(")
        print("    checkpoint_path='/path/to/your/model.checkpoint',")
        print("    observation_variables=['T850', 'T500', 'T300'],")
        print("    num_obs=100,")
        print("    guidance_scale=1.0,")
        print("    use_amip=True,")
        print("    custom_variables=custom_variables  # This fixes the channel mismatch!")
        print(")")
        print("")
        print("# Perform data assimilation")
        print("output, coords = model.sample(batch, seed=42)")
        print("```")
        return
    
    try:
        # Quick setup with custom variables
        print("\nLoading model with custom variable configuration...")
        model, batch = cbottle.quick_regression_guidance_setup(
            checkpoint_path=checkpoint_path,
            observation_variables=['T850', 'T500', 'T300'],
            num_obs=100,
            guidance_scale=1.0,
            use_amip=True,
            custom_variables=custom_variables  # This fixes the channel mismatch!
        )
        
        print(f"\n✅ Model loaded successfully!")
        print(f"  - Model channels: {len(model.batch_info.channels)}")
        print(f"  - Batch target shape: {batch['target'].shape}")
        print(f"  - Expected channels: {len(custom_variables)}")
        
        # Perform data assimilation
        print("\nPerforming data assimilation...")
        output, coords = model.sample(batch, seed=42)
        
        print(f"✅ Data assimilation complete!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Coordinates shape: {coords.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis might be due to:")
        print("1. Incorrect checkpoint path")
        print("2. Model architecture mismatch")
        print("3. Missing dependencies")
        print("4. GPU memory issues")

if __name__ == "__main__":
    main()
