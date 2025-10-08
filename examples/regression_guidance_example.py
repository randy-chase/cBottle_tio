#!/usr/bin/env python3
"""
Example of using regression guidance with cBottle_tio for Score-Based Data Assimilation.

This script demonstrates how to:
1. Load a custom cBottle model
2. Set up regression guidance using ERA5 temperature observations
3. Perform data assimilation by constraining the diffusion sampling
"""

import os
import torch
import numpy as np
from datetime import datetime
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regression_guided_inference import load_custom_model_with_regression_guidance
from regression_guidance import create_observation_data_from_era5


def create_dummy_era5_data(batch_info, num_pixels=1000):
    """
    Create dummy ERA5 data for demonstration.
    
    In practice, you would load real ERA5 data from files.
    """
    print("Creating dummy ERA5 data...")
    
    # Your model's variable keys
    variable_keys = [
        'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
        'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10', 
        'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
        'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
        'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
        'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut', 
        'pres_msl', 'pr', 'rsds', 'sst', 'sic'
    ]
    
    # Create realistic temperature values (in Kelvin)
    num_channels = len(variable_keys)
    time_length = 1  # Single time step for simplicity
    
    era5_data = torch.zeros(num_channels, time_length, num_pixels)
    
    # Set realistic temperature values for different pressure levels
    temp_channels = [i for i, var in enumerate(variable_keys) if var.startswith('T')]
    for i, channel_idx in enumerate(temp_channels):
        pressure_level = variable_keys[channel_idx][1:]  # Extract pressure level
        if pressure_level == '1000':
            base_temp = 288.0  # Surface temperature
        elif pressure_level == '850':
            base_temp = 280.0
        elif pressure_level == '700':
            base_temp = 270.0
        elif pressure_level == '500':
            base_temp = 250.0
        elif pressure_level == '300':
            base_temp = 220.0
        elif pressure_level == '200':
            base_temp = 200.0
        elif pressure_level == '50':
            base_temp = 180.0
        elif pressure_level == '10':
            base_temp = 160.0
        else:
            base_temp = 250.0  # Default
        
        # Add some spatial variation
        spatial_variation = torch.randn(num_pixels) * 10
        era5_data[channel_idx, 0, :] = base_temp + spatial_variation
    
    # Set other variables to reasonable values
    for i, var in enumerate(variable_keys):
        if not var.startswith('T'):
            if var.startswith('U') or var.startswith('V'):
                # Wind components
                era5_data[i, 0, :] = torch.randn(num_pixels) * 5
            elif var.startswith('Z'):
                # Geopotential height
                era5_data[i, 0, :] = torch.randn(num_pixels) * 100 + 5000
            elif var.startswith('Q'):
                # Specific humidity
                era5_data[i, 0, :] = torch.randn(num_pixels) * 0.001 + 0.005
            else:
                # Other variables
                era5_data[i, 0, :] = torch.randn(num_pixels) * 0.1
    
    print(f"Created ERA5 data with shape: {era5_data.shape}")
    return era5_data


def create_observation_locations(num_pixels, num_obs=100):
    """
    Create observation locations (pixel indices where we have observations).
    
    In practice, these would correspond to actual weather station locations
    or satellite observation points.
    """
    # Randomly select observation locations
    obs_locations = torch.randint(0, num_pixels, (num_obs,))
    return obs_locations


def main():
    """
    Main example function demonstrating regression guidance.
    """
    print("Regression Guidance Example for Score-Based Data Assimilation")
    print("=" * 70)
    
    # Configuration
    checkpoint_path = "/path/to/your/custom-model.checkpoint"  # Update this path
    model_name = "my-weather-model"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
        print("For now, we'll demonstrate the setup without loading the model.")
        return
    
    try:
        # Load your custom model with regression guidance support
        print("Loading custom model...")
        model = load_custom_model_with_regression_guidance(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            sigma_min=0.02,
            sigma_max=200.0,
            num_steps=18
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"Available variables: {model.batch_info.channels}")
        
        # Create dummy ERA5 data (replace with real data loading)
        era5_data = create_dummy_era5_data(model.batch_info, num_pixels=1000)
        
        # Create observation locations
        observation_locations = create_observation_locations(num_pixels=1000, num_obs=100)
        
        # Define which variables we want to observe (temperature at different levels)
        observation_variables = ['T850', 'T500', 'T300']  # Temperature at 850, 500, 300 hPa
        
        # Create observation data from ERA5
        observation_data, observation_uncertainty = create_observation_data_from_era5(
            era5_data=era5_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            batch_info=model.batch_info,
            uncertainty=0.1  # 0.1 K uncertainty
        )
        
        print(f"✅ Created observation data:")
        print(f"  - Shape: {observation_data.shape}")
        print(f"  - Variables: {observation_variables}")
        print(f"  - Locations: {len(observation_locations)}")
        
        # Set up regression guidance
        print("Setting up regression guidance...")
        model.set_regression_guidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=0.1,  # 0.1 K uncertainty
            guidance_scale=1.0,  # Moderate guidance strength
        )
        
        # Create a proper batch using AMIP dataset
        print("Loading AMIP dataset for proper batch preparation...")
        try:
            from cbottle.datasets.dataset_3d import get_dataset
            ds = get_dataset(dataset="amip")
            loader = torch.utils.data.DataLoader(ds, batch_size=1)
            batch = next(iter(loader))
            print(f"✅ AMIP batch loaded with shape: {batch['target'].shape}")
            print(f"  - Time info: {batch['second_of_day']} seconds, day {batch['day_of_year']}")
            print(f"  - Has SST condition: {batch['condition'].shape}")
        except Exception as e:
            print(f"⚠️  Could not load AMIP dataset: {e}")
            print("Creating dummy batch instead...")
            batch_size = 1
            time_length = model.time_length
            num_channels = len(model.batch_info.channels)
            num_pixels = model.net.domain.numel()
            
            batch = {
                "target": torch.randn(batch_size, num_channels, time_length, num_pixels),
                "labels": torch.zeros(batch_size, 2),  # Assuming 2 dataset types
                "condition": torch.randn(batch_size, 0, time_length, num_pixels),  # No condition channels
                "second_of_day": torch.randint(0, 86400, (batch_size,)),
                "day_of_year": torch.randint(1, 366, (batch_size,))
            }
            print(f"✅ Dummy batch created with shape: {batch['target'].shape}")
        
        # Perform data assimilation with regression guidance
        print("Performing data assimilation with regression guidance...")
        output, coords = model.sample(batch, seed=42)
        
        print(f"✅ Data assimilation complete!")
        print(f"Output shape: {output.shape}")
        print(f"Coordinates: {coords}")
        
        # You can now analyze the results
        print("\nAnalysis:")
        print("- The generated samples should be consistent with the observed temperatures")
        print("- The regression guidance constrains the diffusion process to match observations")
        print("- This is a form of score-based data assimilation")
        
        # Example: Check if temperature values are reasonable
        temp_channels = [i for i, var in enumerate(model.batch_info.channels) if var.startswith('T')]
        if temp_channels:
            sample_temps = output[0, temp_channels, 0, :].mean(dim=0)  # Average over pressure levels
            print(f"Sample temperature range: {sample_temps.min():.1f}K - {sample_temps.max():.1f}K")
        
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to model compatibility issues or missing dependencies.")


def demonstrate_without_model():
    """
    Demonstrate the setup without actually loading a model.
    """
    print("\nDemonstration without model loading:")
    print("-" * 40)
    
    # Create dummy data to show the structure
    num_pixels = 1000
    num_obs = 100
    
    # Dummy observation data
    observation_data = torch.randn(num_obs, 3) * 10 + 273.15  # T850, T500, T300
    observation_locations = torch.randint(0, num_pixels, (num_obs,))
    observation_variables = ['T850', 'T500', 'T300']
    
    print(f"Observation data shape: {observation_data.shape}")
    print(f"Observation locations: {observation_locations.shape}")
    print(f"Variables: {observation_variables}")
    print(f"Temperature range: {observation_data.min():.1f}K - {observation_data.max():.1f}K")
    
    print("\nTo use with your actual model:")
    print("1. Update checkpoint_path in main()")
    print("2. Replace create_dummy_era5_data() with real ERA5 data loading")
    print("3. Adjust observation_variables to match your needs")
    print("4. Tune guidance_scale and observation_uncertainty parameters")


if __name__ == "__main__":
    main()
    demonstrate_without_model()
