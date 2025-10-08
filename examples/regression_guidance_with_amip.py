#!/usr/bin/env python3
"""
Example of using regression guidance with cBottle_tio and the AMIP dataset.

This script demonstrates how to:
1. Load a custom cBottle model
2. Use the real AMIP dataset for proper batch preparation
3. Set up regression guidance using ERA5 temperature observations
4. Perform data assimilation with real atmospheric data
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import pandas as pd

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regression_guided_inference import load_custom_model_with_regression_guidance
from regression_guidance import create_observation_data_from_era5
from cbottle.datasets.dataset_3d import get_dataset


def create_era5_observations_from_amip_batch(
    amip_batch: dict,
    observation_variables: list,
    observation_locations: torch.Tensor,
    batch_info,
) -> torch.Tensor:
    """
    Create ERA5 observations from AMIP batch data.
    
    This function extracts the observed variables from the AMIP batch
    at the specified observation locations to create the observation data
    for regression guidance.
    
    Args:
        amip_batch: Batch from AMIP dataset
        observation_variables: List of variables to observe (e.g., ['T850', 'T500'])
        observation_locations: Pixel indices where observations are available
        batch_info: cBottle batch info for variable mapping
        
    Returns:
        observation_data: [num_obs, num_variables] tensor
    """
    print("Creating ERA5 observations from AMIP batch...")
    
    # Get the target data from AMIP batch
    target = amip_batch["target"]  # [batch, channels, time, pixels]
    
    # Map variable names to channel indices
    variable_to_channel = {var: i for i, var in enumerate(batch_info.channels)}
    observed_channels = [variable_to_channel[var] for var in observation_variables]
    
    # Extract observations at specified locations
    num_obs = len(observation_locations)
    num_vars = len(observation_variables)
    
    observation_data = torch.zeros(num_obs, num_vars, device=target.device, dtype=target.dtype)
    
    for i, channel_idx in enumerate(observed_channels):
        # Extract values for this variable at observation locations
        # target: [batch, channels, time, pixels]
        # observation_locations: [num_obs]
        var_data = target[0, channel_idx, :, observation_locations]  # [time, num_obs]
        
        # Average over time dimension (or take first time step)
        observation_data[:, i] = var_data.mean(dim=0)  # [num_obs]
        
        print(f"Extracted {observation_variables[i]} (channel {channel_idx}): "
              f"mean={observation_data[:, i].mean():.2f}, "
              f"std={observation_data[:, i].std():.2f}")
    
    print(f"Created observation data with shape: {observation_data.shape}")
    return observation_data


def create_observation_locations_from_amip(
    amip_batch: dict,
    num_obs: int = 100,
    strategy: str = "random"
) -> torch.Tensor:
    """
    Create observation locations from AMIP batch.
    
    Args:
        amip_batch: Batch from AMIP dataset
        num_obs: Number of observations to create
        strategy: Strategy for selecting locations ("random", "uniform", "land_only")
        
    Returns:
        observation_locations: [num_obs] tensor of pixel indices
    """
    target = amip_batch["target"]
    num_pixels = target.shape[-1]  # Last dimension is pixels
    
    if strategy == "random":
        # Randomly select observation locations
        obs_locations = torch.randint(0, num_pixels, (num_obs,))
    elif strategy == "uniform":
        # Uniformly distributed observations
        step = num_pixels // num_obs
        obs_locations = torch.arange(0, num_pixels, step)[:num_obs]
    elif strategy == "land_only":
        # Select only land pixels (where SST is not available)
        # This is a simplified approach - in practice you'd use a proper land mask
        sst_channel = None
        for i, var in enumerate(amip_batch.get("batch_info", {}).get("channels", [])):
            if var == "sst":
                sst_channel = i
                break
        
        if sst_channel is not None:
            # Find pixels where SST is NaN (land)
            sst_data = target[0, sst_channel, 0, :]  # [pixels]
            land_mask = torch.isnan(sst_data)
            land_pixels = torch.where(land_mask)[0]
            
            if len(land_pixels) >= num_obs:
                # Randomly select from land pixels
                indices = torch.randperm(len(land_pixels))[:num_obs]
                obs_locations = land_pixels[indices]
            else:
                # Not enough land pixels, use all land pixels + random ocean pixels
                obs_locations = land_pixels
                remaining = num_obs - len(land_pixels)
                ocean_pixels = torch.where(~land_mask)[0]
                if len(ocean_pixels) >= remaining:
                    indices = torch.randperm(len(ocean_pixels))[:remaining]
                    obs_locations = torch.cat([obs_locations, ocean_pixels[indices]])
                else:
                    obs_locations = torch.cat([obs_locations, ocean_pixels])
        else:
            # Fallback to random if SST channel not found
            obs_locations = torch.randint(0, num_pixels, (num_obs,))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Created {len(obs_locations)} observation locations using '{strategy}' strategy")
    return obs_locations


def main():
    """
    Main example function demonstrating regression guidance with AMIP dataset.
    """
    print("Regression Guidance with AMIP Dataset Example")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = "/path/to/your/custom-model.checkpoint"  # Update this path
    model_name = "my-weather-model"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
        print("For now, we'll demonstrate the setup with the AMIP dataset.")
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
        
        # Load AMIP dataset
        print("Loading AMIP dataset...")
        ds = get_dataset(dataset="amip")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        amip_batch = next(iter(loader))
        
        print(f"✅ AMIP batch loaded successfully!")
        print(f"Batch keys: {list(amip_batch.keys())}")
        print(f"Target shape: {amip_batch['target'].shape}")
        print(f"Condition shape: {amip_batch['condition'].shape}")
        print(f"Labels shape: {amip_batch['labels'].shape}")
        print(f"Second of day: {amip_batch['second_of_day']}")
        print(f"Day of year: {amip_batch['day_of_year']}")
        
        # Create observation locations
        observation_locations = create_observation_locations_from_amip(
            amip_batch=amip_batch,
            num_obs=100,
            strategy="random"  # or "uniform" or "land_only"
        )
        
        # Define which variables we want to observe
        observation_variables = ['T850', 'T500', 'T300']  # Temperature at different pressure levels
        
        # Create observation data from AMIP batch
        observation_data = create_era5_observations_from_amip_batch(
            amip_batch=amip_batch,
            observation_variables=observation_variables,
            observation_locations=observation_locations,
            batch_info=model.batch_info
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
        
        # Perform data assimilation with regression guidance
        print("Performing data assimilation with regression guidance...")
        output, coords = model.sample(amip_batch, seed=42)
        
        print(f"✅ Data assimilation complete!")
        print(f"Output shape: {output.shape}")
        print(f"Coordinates: {coords}")
        
        # Analysis
        print("\nAnalysis:")
        print("- The generated samples are constrained by the observed temperatures")
        print("- The regression guidance ensures consistency with observations")
        print("- This demonstrates score-based data assimilation with real atmospheric data")
        
        # Compare original vs guided output
        original_target = amip_batch["target"][0]  # [channels, time, pixels]
        guided_output = output[0]  # [channels, time, pixels]
        
        # Check temperature consistency at observation locations
        temp_channels = [i for i, var in enumerate(model.batch_info.channels) 
                        if var in observation_variables]
        
        if temp_channels:
            print(f"\nTemperature consistency check:")
            for i, var in enumerate(observation_variables):
                channel_idx = model.batch_info.channels.index(var)
                obs_values = observation_data[:, i]
                guided_values = guided_output[channel_idx, 0, observation_locations]
                original_values = original_target[channel_idx, 0, observation_locations]
                
                obs_guided_diff = torch.abs(obs_values - guided_values).mean()
                obs_original_diff = torch.abs(obs_values - original_values).mean()
                
                print(f"  {var}:")
                print(f"    Obs vs Guided diff: {obs_guided_diff:.3f}K")
                print(f"    Obs vs Original diff: {obs_original_diff:.3f}K")
                print(f"    Improvement: {((obs_original_diff - obs_guided_diff) / obs_original_diff * 100):.1f}%")
        
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual model path.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_amip_batch_structure():
    """
    Demonstrate the structure of AMIP batch data.
    """
    print("\nAMIP Batch Structure Demonstration:")
    print("-" * 40)
    
    try:
        # Load AMIP dataset
        ds = get_dataset(dataset="amip")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        
        print("AMIP batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        # Show available variables
        if hasattr(ds, 'batch_info'):
            print(f"\nAvailable variables ({len(ds.batch_info.channels)}):")
            for i, var in enumerate(ds.batch_info.channels):
                print(f"  {i:2d}: {var}")
        
        # Show time information
        if 'timestamp' in batch:
            timestamp = batch['timestamp']
            print(f"\nTimestamp: {timestamp}")
        
        if 'second_of_day' in batch:
            second_of_day = batch['second_of_day'].item()
            hours = second_of_day / 3600
            print(f"Time of day: {hours:.1f} hours")
        
        if 'day_of_year' in batch:
            day_of_year = batch['day_of_year'].item()
            print(f"Day of year: {day_of_year:.0f}")
        
    except Exception as e:
        print(f"❌ Error loading AMIP dataset: {e}")
        print("Make sure the AMIP dataset is properly configured.")


def create_synthetic_observations_example():
    """
    Example of creating synthetic observations for testing.
    """
    print("\nSynthetic Observations Example:")
    print("-" * 35)
    
    try:
        # Load AMIP dataset
        ds = get_dataset(dataset="amip")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        
        # Create synthetic observations
        num_obs = 50
        observation_variables = ['T850', 'T500', 'T300']
        
        # Create observation locations
        target = batch["target"]
        num_pixels = target.shape[-1]
        observation_locations = torch.randint(0, num_pixels, (num_obs,))
        
        # Create synthetic observation data
        observation_data = torch.zeros(num_obs, len(observation_variables))
        
        # Set realistic temperature values
        base_temps = [280.0, 250.0, 220.0]  # T850, T500, T300 in Kelvin
        for i, (var, base_temp) in enumerate(zip(observation_variables, base_temps)):
            # Add some realistic variation
            variation = torch.randn(num_obs) * 5  # ±5K variation
            observation_data[:, i] = base_temp + variation
        
        print(f"Created synthetic observations:")
        print(f"  - Shape: {observation_data.shape}")
        print(f"  - Variables: {observation_variables}")
        print(f"  - Temperature ranges:")
        for i, var in enumerate(observation_variables):
            min_temp = observation_data[:, i].min()
            max_temp = observation_data[:, i].max()
            mean_temp = observation_data[:, i].mean()
            print(f"    {var}: {min_temp:.1f}K - {max_temp:.1f}K (mean: {mean_temp:.1f}K)")
        
        return observation_data, observation_locations, observation_variables
        
    except Exception as e:
        print(f"❌ Error creating synthetic observations: {e}")
        return None, None, None


if __name__ == "__main__":
    main()
    demonstrate_amip_batch_structure()
    create_synthetic_observations_example()
