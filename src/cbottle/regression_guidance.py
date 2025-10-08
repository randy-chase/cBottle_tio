#!/usr/bin/env python3
"""
Regression guidance for cBottle_tio using weather observations.

This module provides regression-based guidance that constrains the diffusion
sampling process to match observed weather data, similar to the approach
used in the appa repository but adapted for cBottle_tio's architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
from functools import partial


class RegressionGuidance:
    """
    Regression guidance system that constrains diffusion sampling to match observations.
    
    This replaces the classifier-based guidance in cBottle_tio with a physics-based
    approach that minimizes the difference between predicted and observed values.
    """
    
    def __init__(
        self,
        observation_data: torch.Tensor,
        observation_locations: torch.Tensor,  # Pixel indices where observations are available
        observation_variables: List[str],     # Which variables are observed (e.g., ['T850', 'T500'])
        observation_uncertainty: float = 0.1, # Standard deviation of observation errors
        guidance_scale: float = 1.0,          # Strength of guidance
        batch_info: Optional[object] = None,  # cBottle batch info for variable mapping
    ):
        """
        Initialize regression guidance.
        
        Args:
            observation_data: Observed values [num_obs, num_variables]
            observation_locations: Pixel indices [num_obs] where observations are available
            observation_variables: List of variable names being observed
            observation_uncertainty: Standard deviation of observation errors
            guidance_scale: Strength of guidance (higher = stronger constraint)
            batch_info: cBottle batch info for variable name mapping
        """
        self.observation_data = observation_data.cuda() if observation_data.device.type == 'cpu' else observation_data
        self.observation_locations = observation_locations.cuda() if observation_locations.device.type == 'cpu' else observation_locations
        self.observation_variables = observation_variables
        self.observation_uncertainty = observation_uncertainty
        self.guidance_scale = guidance_scale
        self.batch_info = batch_info
        
        # Map variable names to channel indices
        if batch_info is not None:
            self.variable_to_channel = {var: i for i, var in enumerate(batch_info.channels)}
            self.observed_channels = [self.variable_to_channel[var] for var in observation_variables]
        else:
            # Fallback: assume variables are in the order they appear in the list
            self.observed_channels = list(range(len(observation_variables)))
        
        print(f"Regression guidance initialized:")
        print(f"  - {len(observation_locations)} observation locations")
        print(f"  - Variables: {observation_variables}")
        print(f"  - Channels: {self.observed_channels}")
        print(f"  - Uncertainty: {observation_uncertainty}")
        print(f"  - Guidance scale: {guidance_scale}")
    
    def create_forward_operator(self, model_output_shape: Tuple[int, ...]) -> Callable:
        """
        Create a forward operator that extracts observed variables from model output.
        
        Args:
            model_output_shape: Shape of model output [batch, channels, time, pixels]
            
        Returns:
            Forward operator function A(x) -> y
        """
        batch_size, num_channels, time_length, num_pixels = model_output_shape
        
        def forward_operator(x: torch.Tensor) -> torch.Tensor:
            """
            Extract observed variables at observation locations.
            
            Args:
                x: Model output [batch, channels, time, pixels]
                
            Returns:
                y: Extracted observations [batch, num_obs, num_variables]
            """
            # Ensure x is on the right device
            x = x.to(self.observation_data.device)
            
            # Extract values at observation locations for observed channels
            # x: [batch, channels, time, pixels]
            # observation_locations: [num_obs]
            # observed_channels: [num_variables]
            
            batch_size = x.shape[0]
            num_obs = len(self.observation_locations)
            num_vars = len(self.observed_channels)
            
            # Extract values: [batch, num_obs, num_variables, time]
            extracted = torch.zeros(batch_size, num_obs, num_vars, time_length, 
                                  device=x.device, dtype=x.dtype)
            
            for i, channel_idx in enumerate(self.observed_channels):
                # Extract values for this variable at all observation locations
                # x[:, channel_idx, :, observation_locations] -> [batch, time, num_obs]
                extracted[:, :, i, :] = x[:, channel_idx, :, self.observation_locations].transpose(1, 2)
            
            # Average over time dimension for now (could be made configurable)
            extracted = extracted.mean(dim=3)  # [batch, num_obs, num_variables]
            
            return extracted
        
        return forward_operator
    
    def compute_guidance_gradient(
        self, 
        x_hat: torch.Tensor, 
        denoised: torch.Tensor, 
        t_hat: torch.Tensor,
        forward_operator: Callable
    ) -> torch.Tensor:
        """
        Compute guidance gradient using regression loss.
        
        Args:
            x_hat: Current noisy state [batch, channels, time, pixels]
            denoised: Denoised prediction [batch, channels, time, pixels]
            t_hat: Current noise level
            forward_operator: Forward operator A(x) -> y
            
        Returns:
            Guidance gradient [batch, channels, time, pixels]
        """
        # Ensure gradients are enabled
        x_hat = x_hat.detach().requires_grad_(True)
        
        # Get predicted observations
        y_pred = forward_operator(x_hat)  # [batch, num_obs, num_variables]
        
        # Expand observation data to match batch size
        y_obs = self.observation_data.unsqueeze(0).expand(x_hat.shape[0], -1, -1)
        
        # Compute regression loss (MSE with uncertainty weighting)
        residual = y_pred - y_obs  # [batch, num_obs, num_variables]
        loss = (residual ** 2).sum() / (2 * self.observation_uncertainty ** 2)
        
        # Compute gradient
        guidance_grad = torch.autograd.grad(loss, x_hat, retain_graph=False)[0]
        
        # Normalize gradient to match score function scale
        score = (x_hat - denoised) / t_hat
        score_norm = torch.norm(score)
        guidance_norm = torch.norm(guidance_grad)
        
        if guidance_norm > 0:
            scale = score_norm / guidance_norm
        else:
            scale = 0.0
        
        # Apply guidance
        return -t_hat * scale * guidance_grad * self.guidance_scale


def create_regression_guided_denoiser(
    net: nn.Module,
    regression_guidance: RegressionGuidance,
    *,
    second_of_day: torch.Tensor,
    day_of_year: torch.Tensor,
    labels: torch.Tensor,
    condition: torch.Tensor,
) -> Callable:
    """
    Create a denoiser with regression guidance.
    
    This replaces the classifier-based guidance in cBottle_tio with regression guidance.
    
    Args:
        net: Base diffusion network
        regression_guidance: Regression guidance system
        second_of_day: Time of day tensor
        day_of_year: Day of year tensor  
        labels: Class labels
        condition: Conditioning data
        
    Returns:
        Guided denoiser function D(x_hat, t_hat) -> denoised
    """
    
    def guided_denoiser(x_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        """
        Denoiser with regression guidance.
        
        Args:
            x_hat: Noisy input [batch, channels, time, pixels]
            t_hat: Noise level
            
        Returns:
            Denoised output with guidance applied
        """
        # Enable gradients for guidance
        x_hat = x_hat.detach().requires_grad_(True)
        
        # Get base denoised prediction
        out = net(
            x_hat,
            t_hat,
            class_labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
        )
        
        denoised = out.out
        
        # Apply regression guidance
        forward_operator = regression_guidance.create_forward_operator(x_hat.shape)
        guidance_grad = regression_guidance.compute_guidance_gradient(
            x_hat, denoised, t_hat, forward_operator
        )
        
        # Add guidance to denoised output
        guided_denoised = denoised + guidance_grad
        
        return guided_denoised
    
    # Set required attributes for EDM sampler
    guided_denoiser.round_sigma = net.round_sigma
    guided_denoiser.sigma_max = net.sigma_max
    guided_denoiser.sigma_min = net.sigma_min
    
    return guided_denoiser


def create_observation_data_from_era5(
    era5_data: torch.Tensor,
    observation_locations: torch.Tensor,
    observation_variables: List[str],
    batch_info: object,
    uncertainty: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create observation data from ERA5 for regression guidance.
    
    Args:
        era5_data: ERA5 data [channels, time, pixels] or [batch, channels, time, pixels]
        observation_locations: Pixel indices where observations are available
        observation_variables: List of variable names to observe
        batch_info: cBottle batch info for variable mapping
        uncertainty: Observation uncertainty
        
    Returns:
        observation_data: [num_obs, num_variables]
        observation_uncertainty: [num_obs, num_variables]
    """
    # Handle different input shapes
    if era5_data.dim() == 3:
        era5_data = era5_data.unsqueeze(0)  # Add batch dimension
    
    # Map variable names to channel indices
    variable_to_channel = {var: i for i, var in enumerate(batch_info.channels)}
    observed_channels = [variable_to_channel[var] for var in observation_variables]
    
    # Extract observations
    num_obs = len(observation_locations)
    num_vars = len(observation_variables)
    
    # Extract values at observation locations
    # era5_data: [batch, channels, time, pixels]
    # observation_locations: [num_obs]
    # observed_channels: [num_variables]
    
    observations = torch.zeros(num_obs, num_vars, device=era5_data.device, dtype=era5_data.dtype)
    
    for i, channel_idx in enumerate(observed_channels):
        # Extract values for this variable at all observation locations
        # Average over batch and time dimensions
        var_data = era5_data[:, channel_idx, :, observation_locations]  # [batch, time, num_obs]
        observations[:, i] = var_data.mean(dim=(0, 1))  # [num_obs]
    
    # Create uncertainty tensor
    uncertainties = torch.full((num_obs, num_vars), uncertainty, 
                             device=era5_data.device, dtype=era5_data.dtype)
    
    return observations, uncertainties


# Example usage function
def example_regression_guidance():
    """
    Example of how to use regression guidance with cBottle_tio.
    """
    print("Regression Guidance Example")
    print("=" * 40)
    
    # This would be used in your custom model loading script
    # model = load_custom_model("path/to/your/model.checkpoint")
    
    # Create dummy observation data (replace with real ERA5 data)
    num_obs = 100
    num_vars = 2  # T850, T500
    observation_data = torch.randn(num_obs, num_vars) * 10 + 273.15  # Temperature in Kelvin
    observation_locations = torch.randint(0, 1000, (num_obs,))  # Random pixel locations
    observation_variables = ['T850', 'T500']
    
    # Create regression guidance
    guidance = RegressionGuidance(
        observation_data=observation_data,
        observation_locations=observation_locations,
        observation_variables=observation_variables,
        observation_uncertainty=0.1,
        guidance_scale=1.0,
    )
    
    print("Regression guidance created successfully!")
    print(f"Observation data shape: {observation_data.shape}")
    print(f"Observation locations shape: {observation_locations.shape}")
    
    return guidance


if __name__ == "__main__":
    example_regression_guidance()
