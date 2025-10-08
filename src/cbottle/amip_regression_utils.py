#!/usr/bin/env python3
"""
Utility functions for working with AMIP dataset and regression guidance.

This module provides helper functions to easily set up regression guidance
with the AMIP dataset for score-based data assimilation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from .datasets.dataset_3d import get_dataset
from .regression_guided_inference import load_custom_model_with_regression_guidance


def load_amip_batch(batch_size: int = 1) -> Tuple[dict, object]:
    """
    Load a batch from the AMIP dataset.
    
    Args:
        batch_size: Batch size for the data loader
        
    Returns:
        batch: AMIP batch dictionary
        dataset: AMIP dataset object
    """
    print("Loading AMIP dataset...")
    ds = get_dataset(dataset="amip")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    batch = next(iter(loader))
    
    print(f"✅ AMIP batch loaded:")
    print(f"  - Target shape: {batch['target'].shape}")
    print(f"  - Condition shape: {batch['condition'].shape}")
    print(f"  - Labels shape: {batch['labels'].shape}")
    print(f"  - Time: {batch['second_of_day']} seconds, day {batch['day_of_year']}")
    
    return batch, ds


def setup_regression_guidance_with_amip(
    model,
    amip_batch: dict,
    observation_variables: List[str],
    num_obs: int = 100,
    observation_strategy: str = "random",
    observation_uncertainty: float = 0.1,
    guidance_scale: float = 1.0,
) -> None:
    """
    Set up regression guidance using AMIP batch data.
    
    Args:
        model: RegressionGuidedCBottle3d model
        amip_batch: Batch from AMIP dataset
        observation_variables: List of variables to observe (e.g., ['T850', 'T500'])
        num_obs: Number of observations to create
        observation_strategy: Strategy for selecting locations ("random", "uniform", "land_only")
        observation_uncertainty: Standard deviation of observation errors
        guidance_scale: Strength of guidance
    """
    print(f"Setting up regression guidance with {num_obs} observations...")
    
    # Create observation locations
    observation_locations = create_observation_locations(
        amip_batch=amip_batch,
        num_obs=num_obs,
        strategy=observation_strategy
    )
    
    # Create observation data from AMIP batch
    observation_data = create_observations_from_amip(
        amip_batch=amip_batch,
        observation_variables=observation_variables,
        observation_locations=observation_locations,
        batch_info=model.batch_info
    )
    
    # Set up regression guidance
    model.set_regression_guidance(
        observation_data=observation_data,
        observation_locations=observation_locations,
        observation_variables=observation_variables,
        observation_uncertainty=observation_uncertainty,
        guidance_scale=guidance_scale,
    )
    
    print(f"✅ Regression guidance configured:")
    print(f"  - Variables: {observation_variables}")
    print(f"  - Locations: {len(observation_locations)}")
    print(f"  - Uncertainty: {observation_uncertainty}")
    print(f"  - Guidance scale: {guidance_scale}")


def create_observation_locations(
    amip_batch: dict,
    num_obs: int = 100,
    strategy: str = "random"
) -> torch.Tensor:
    """
    Create observation locations from AMIP batch.
    
    Args:
        amip_batch: Batch from AMIP dataset
        num_obs: Number of observations to create
        strategy: Strategy for selecting locations
        
    Returns:
        observation_locations: [num_obs] tensor of pixel indices
    """
    target = amip_batch["target"]
    num_pixels = target.shape[-1]  # Last dimension is pixels
    
    if strategy == "random":
        obs_locations = torch.randint(0, num_pixels, (num_obs,))
    elif strategy == "uniform":
        step = num_pixels // num_obs
        obs_locations = torch.arange(0, num_pixels, step)[:num_obs]
    elif strategy == "land_only":
        # Try to find land pixels using SST data
        sst_channel = None
        for i, var in enumerate(amip_batch.get("batch_info", {}).get("channels", [])):
            if var == "sst":
                sst_channel = i
                break
        
        if sst_channel is not None:
            sst_data = target[0, sst_channel, 0, :]
            land_mask = torch.isnan(sst_data)
            land_pixels = torch.where(land_mask)[0]
            
            if len(land_pixels) >= num_obs:
                indices = torch.randperm(len(land_pixels))[:num_obs]
                obs_locations = land_pixels[indices]
            else:
                obs_locations = land_pixels
                remaining = num_obs - len(land_pixels)
                ocean_pixels = torch.where(~land_mask)[0]
                if len(ocean_pixels) >= remaining:
                    indices = torch.randperm(len(ocean_pixels))[:remaining]
                    obs_locations = torch.cat([obs_locations, ocean_pixels[indices]])
                else:
                    obs_locations = torch.cat([obs_locations, ocean_pixels])
        else:
            obs_locations = torch.randint(0, num_pixels, (num_obs,))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Created {len(obs_locations)} observation locations using '{strategy}' strategy")
    return obs_locations


def create_observations_from_amip(
    amip_batch: dict,
    observation_variables: List[str],
    observation_locations: torch.Tensor,
    batch_info,
) -> torch.Tensor:
    """
    Create observation data from AMIP batch.
    
    Args:
        amip_batch: Batch from AMIP dataset
        observation_variables: List of variables to observe
        observation_locations: Pixel indices where observations are available
        batch_info: cBottle batch info for variable mapping
        
    Returns:
        observation_data: [num_obs, num_variables] tensor
    """
    target = amip_batch["target"]  # [batch, channels, time, pixels]
    
    # Map variable names to channel indices
    variable_to_channel = {var: i for i, var in enumerate(batch_info.channels)}
    observed_channels = [variable_to_channel[var] for var in observation_variables]
    
    # Extract observations at specified locations
    num_obs = len(observation_locations)
    num_vars = len(observation_variables)
    
    observation_data = torch.zeros(num_obs, num_vars, device=target.device, dtype=target.dtype)
    
    for i, channel_idx in enumerate(observed_channels):
        var_data = target[0, channel_idx, :, observation_locations]  # [time, num_obs]
        observation_data[:, i] = var_data.mean(dim=0)  # Average over time
        
        print(f"  {observation_variables[i]}: "
              f"mean={observation_data[:, i].mean():.2f}, "
              f"std={observation_data[:, i].std():.2f}")
    
    return observation_data


def create_synthetic_observations(
    observation_variables: List[str],
    num_obs: int = 100,
    base_temperatures: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic observation data for testing.
    
    Args:
        observation_variables: List of variables to observe
        num_obs: Number of observations to create
        base_temperatures: Base temperatures for each variable (in Kelvin)
        
    Returns:
        observation_data: [num_obs, num_variables] tensor
        observation_locations: [num_obs] tensor of pixel indices
    """
    if base_temperatures is None:
        # Default base temperatures for different pressure levels
        base_temperatures = {
            'T1000': 288.0,  # Surface
            'T850': 280.0,   # 850 hPa
            'T700': 270.0,   # 700 hPa
            'T500': 250.0,   # 500 hPa
            'T300': 220.0,   # 300 hPa
            'T200': 200.0,   # 200 hPa
            'T50': 180.0,    # 50 hPa
            'T10': 160.0,    # 10 hPa
        }
    
    num_vars = len(observation_variables)
    observation_data = torch.zeros(num_obs, num_vars)
    observation_locations = torch.randint(0, 1000, (num_obs,))  # Random locations
    
    for i, var in enumerate(observation_variables):
        base_temp = base_temperatures.get(var, 250.0)  # Default to 250K
        variation = torch.randn(num_obs) * 5  # ±5K variation
        observation_data[:, i] = base_temp + variation
    
    print(f"Created synthetic observations:")
    print(f"  - Shape: {observation_data.shape}")
    print(f"  - Variables: {observation_variables}")
    
    return observation_data, observation_locations


def create_custom_batch_info(variables: List[str]) -> 'BatchInfo':
    """
    Create a BatchInfo object for a custom model with the specified variables.
    
    Args:
        variables: List of variable names in the order they appear in the model
        
    Returns:
        BatchInfo object with the correct channel mapping
    """
    from .datasets.base import BatchInfo, TimeUnit
    
    return BatchInfo(
        channels=variables,
        time_step=1,
        time_unit=TimeUnit.HOUR
    )


def create_synthetic_batch_with_channels(num_channels: int, condition_channels: int = 0) -> dict:
    """
    Create a synthetic batch with the specified number of channels.
    
    Args:
        num_channels: Number of channels in the target
        condition_channels: Number of channels in the condition (0 for no conditioning)
        
    Returns:
        Synthetic batch dictionary
    """
    # Create synthetic data with the correct number of channels
    batch = {
        'target': torch.randn(1, num_channels, 1, 12288),  # Standard cBottle shape
        'condition': torch.randn(1, condition_channels, 1, 12288) if condition_channels > 0 else torch.zeros(1, 0, 1, 12288),  # Condition tensor
        'labels': torch.zeros(1, 0),  # No labels
        'second_of_day': torch.tensor([43200]),  # Noon
        'day_of_year': torch.tensor([180]),  # Mid-year
    }
    
    print(f"Created synthetic batch with {num_channels} target channels and {condition_channels} condition channels")
    return batch


def load_custom_model_with_custom_batch_info(
    checkpoint_path: str,
    custom_variables: List[str],
    model_name: Optional[str] = None,
    sigma_min: float = 0.02,
    sigma_max: float = 200.0,
    num_steps: int = 18,
    allow_second_order_derivatives: bool = False,
    **kwargs
) -> 'RegressionGuidedCBottle3d':
    """
    Load a custom model with custom batch_info to handle variable configuration mismatches.
    
    Args:
        checkpoint_path: Path to your custom checkpoint file (.checkpoint)
        custom_variables: List of variable names in the order they appear in the model
        model_name: Optional name for the model (for logging)
        sigma_min: Minimum noise sigma for diffusion sampling
        sigma_max: Maximum noise sigma for diffusion sampling  
        num_steps: Number of sampling steps
        allow_second_order_derivatives: Whether to allow second order derivatives
        **kwargs: Additional arguments passed to CBottle3d
        
    Returns:
        RegressionGuidedCBottle3d: Loaded model with custom batch_info
    """
    import os
    from . import checkpointing
    from .regression_guided_inference import RegressionGuidedCBottle3d
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the model using the checkpointing system
    with checkpointing.Checkpoint(checkpoint_path) as c:
        # Read the model configuration and create the model
        model = c.read_model(
            map_location=None,
            allow_second_order_derivatives=allow_second_order_derivatives
        ).cuda().eval()
        
        # Get the model configuration to determine condition_channels
        model_config = c.read_model_config()
        condition_channels = getattr(model_config, 'condition_channels', 0)
        
        # Create custom batch_info with the specified variables
        custom_batch_info = create_custom_batch_info(custom_variables)
        
        # Attach custom batch_info to the model
        model.batch_info = custom_batch_info
        
        # Store condition_channels for later use
        model.condition_channels = condition_channels
    
    # Create regression-guided version
    regression_model = RegressionGuidedCBottle3d(
        net=model,
        separate_classifier=None,  # No separate classifier for regression guidance
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        **kwargs
    )
    
    print(f"Loaded custom model with {len(custom_variables)} variables: {model_name or 'unnamed'}")
    return regression_model


def quick_regression_guidance_setup(
    checkpoint_path: str,
    observation_variables: List[str] = ['T850', 'T500', 'T300'],
    num_obs: int = 100,
    guidance_scale: float = 1.0,
    use_amip: bool = True,
    custom_variables: Optional[List[str]] = None
) -> Tuple[object, dict]:
    """
    Quick setup for regression guidance with minimal code.
    
    Args:
        checkpoint_path: Path to your model checkpoint
        observation_variables: Variables to observe
        num_obs: Number of observations
        guidance_scale: Guidance strength
        use_amip: Whether to use AMIP dataset or synthetic data
        custom_variables: List of all variables in your custom model (if different from default)
        
    Returns:
        model: Loaded model with regression guidance configured
        batch: Batch data for sampling
    """
    print("Quick regression guidance setup...")
    
    # Load model
    if custom_variables is not None:
        # Use custom batch_info loading for custom variables
        model = load_custom_model_with_custom_batch_info(
            checkpoint_path=checkpoint_path,
            custom_variables=custom_variables,
            model_name="weather-model"
        )
    else:
        # Use standard loading for default variables
        model = load_custom_model_with_regression_guidance(
            checkpoint_path=checkpoint_path,
            model_name="weather-model"
        )
    
    if use_amip:
        # Use AMIP dataset
        batch, ds = load_amip_batch()
        
        # If custom variables are provided, create a synthetic batch with correct channels
        if custom_variables is not None:
            print(f"Using custom variable configuration with {len(custom_variables)} channels")
            print(f"Variables: {custom_variables}")
            
            # Get condition_channels from the model
            condition_channels = getattr(model.net, 'condition_channels', 0)
            print(f"Model expects {condition_channels} condition channels")
            
            # Create a synthetic batch with the correct number of channels
            batch = create_synthetic_batch_with_channels(len(custom_variables), condition_channels)
        else:
            print(f"Using default AMIP configuration with {len(model.batch_info.channels)} channels")
        
        setup_regression_guidance_with_amip(
            model=model,
            amip_batch=batch,
            observation_variables=observation_variables,
            num_obs=num_obs,
            guidance_scale=guidance_scale
        )
    else:
        # Use synthetic data
        if custom_variables is not None:
            # Get condition_channels from the model
            condition_channels = getattr(model.net, 'condition_channels', 0)
            print(f"Model expects {condition_channels} condition channels")
            
            # Create synthetic batch with custom number of channels
            batch = create_synthetic_batch_with_channels(len(custom_variables), condition_channels)
        else:
            # Create synthetic batch with default channels
            batch_size = 1
            time_length = model.time_length
            num_channels = len(model.batch_info.channels)
            num_pixels = model.net.domain.numel()
            condition_channels = getattr(model.net, 'condition_channels', 0)
            
            batch = {
                "target": torch.randn(batch_size, num_channels, time_length, num_pixels),
                "labels": torch.zeros(batch_size, 2),
                "condition": torch.randn(batch_size, condition_channels, time_length, num_pixels) if condition_channels > 0 else torch.zeros(batch_size, 0, time_length, num_pixels),
                "second_of_day": torch.randint(0, 86400, (batch_size,)),
                "day_of_year": torch.randint(1, 366, (batch_size,))
            }
        
        observation_data, observation_locations = create_synthetic_observations(
            observation_variables=observation_variables,
            num_obs=num_obs
        )
        
        model.set_regression_guidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=0.1,
            guidance_scale=guidance_scale,
        )
    
    print("✅ Quick setup complete!")
    return model, batch


# Example usage
def example_quick_setup():
    """
    Example of using the quick setup function.
    """
    print("Quick Setup Example")
    print("=" * 20)
    
    checkpoint_path = "/path/to/your/model.checkpoint"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Please update the path and try again.")
        return
    
    try:
        # Quick setup with AMIP dataset
        model, batch = quick_regression_guidance_setup(
            checkpoint_path=checkpoint_path,
            observation_variables=['T850', 'T500', 'T300'],
            num_obs=100,
            guidance_scale=1.0,
            use_amip=True
        )
        
        # Perform data assimilation
        print("Performing data assimilation...")
        output, coords = model.sample(batch, seed=42)
        
        print(f"✅ Data assimilation complete! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    example_quick_setup()
