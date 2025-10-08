#!/usr/bin/env python3
"""
Modified cBottle_tio inference with regression guidance.

This module extends the CBottle3d class to support regression guidance
using weather observations instead of classifier-based guidance.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import cbottle.inference
import cbottle.diffusion_samplers
from cbottle.diffusion_samplers import edm_sampler_from_sigma
from regression_guidance import RegressionGuidance, create_regression_guided_denoiser


class RegressionGuidedCBottle3d(cbottle.inference.CBottle3d):
    """
    Extended CBottle3d with regression guidance support.
    
    This class replaces the classifier-based guidance with regression guidance
    that constrains the diffusion sampling to match weather observations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as CBottle3d."""
        super().__init__(*args, **kwargs)
        self.regression_guidance = None
    
    def set_regression_guidance(
        self,
        observation_data: torch.Tensor,
        observation_locations: torch.Tensor,
        observation_variables: List[str],
        observation_uncertainty: float = 0.1,
        guidance_scale: float = 1.0,
    ):
        """
        Set up regression guidance for sampling.
        
        Args:
            observation_data: Observed values [num_obs, num_variables]
            observation_locations: Pixel indices [num_obs] where observations are available
            observation_variables: List of variable names being observed
            observation_uncertainty: Standard deviation of observation errors
            guidance_scale: Strength of guidance (higher = stronger constraint)
        """
        self.regression_guidance = RegressionGuidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=observation_uncertainty,
            guidance_scale=guidance_scale,
            batch_info=self.batch_info,
        )
        print("Regression guidance configured successfully!")
    
    def sample_with_regression_guidance(
        self,
        batch: Dict[str, torch.Tensor],
        seed: Optional[int] = None,
        start_from_noisy_image: bool = False,
        bf16: bool = True,
    ) -> Tuple[torch.Tensor, object]:
        """
        Sample with regression guidance instead of classifier guidance.
        
        Args:
            batch: Input batch containing target, labels, condition, etc.
            seed: Random seed for reproducibility
            start_from_noisy_image: Whether to start from noisy input
            bf16: Whether to use bfloat16 precision
            
        Returns:
            output: Generated samples
            coords: Coordinate information
        """
        if self.regression_guidance is None:
            raise ValueError("Regression guidance not set. Call set_regression_guidance() first.")
        
        batch = self._move_to_device(batch)
        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].cuda().float()
        day_of_year = batch["day_of_year"].cuda().float()
        batch_size = second_of_day.shape[0]

        label_ind = labels.nonzero()[:, 1]
        mask = torch.stack([self.icon_mask, self.era5_mask]).cuda()[label_ind]  # n, c
        mask = mask[:, :, None, None]

        with torch.no_grad():
            device = condition.device
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            
            # Create initial noise
            xT = torch.randn_like(images).to(device)
            if start_from_noisy_image:
                xT = images + xT * self.sigma_max
            
            # Check for NaN values
            any_nan = torch.any(torch.isnan(images))
            if any_nan:
                labels_when_nan = torch.zeros_like(labels)
            else:
                labels_when_nan = None
            
            # Create regression-guided denoiser
            def D(x_hat, t_hat):
                # Enable gradients for guidance
                x_hat = x_hat.detach().requires_grad_(True)
                
                # Get base denoised prediction
                out = self.net(
                    x_hat.where(mask, 0),
                    t_hat,
                    class_labels=labels,
                    condition=condition,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                )

                if any_nan:
                    d2 = self.net(
                        x_hat,
                        t_hat,
                        class_labels=labels_when_nan,
                        condition=condition,
                        second_of_day=second_of_day,
                        day_of_year=day_of_year,
                    ).out
                else:
                    d2 = 0.0

                d = out.out.where(mask, d2)

                # Apply regression guidance
                forward_operator = self.regression_guidance.create_forward_operator(x_hat.shape)
                guidance_grad = self.regression_guidance.compute_guidance_gradient(
                    x_hat, d, t_hat, forward_operator
                )
                
                # Add guidance to denoised output
                d = d + guidance_grad

                return d

            # Enable gradients for the denoiser
            D = torch.enable_grad(D)

            # Set required attributes for EDM sampler
            D.round_sigma = self.net.round_sigma
            D.sigma_max = self.net.sigma_max
            D.sigma_min = self.net.sigma_min

            # Run EDM sampler with regression guidance
            with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                out = edm_sampler_from_sigma(
                    D,
                    xT,
                    randn_like=torch.randn_like,
                    sigma_min=self.sigma_min,
                    sigma_max=int(self.sigma_max),
                )

            out = self._post_process(out)

            return out, self.coords
    
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        seed: Optional[int] = None,
        start_from_noisy_image: bool = False,
        guidance_pixels: Optional[torch.Tensor] = None,  # Ignored for regression guidance
        guidance_scale: float = 0.03,  # Ignored for regression guidance
        bf16: bool = True,
    ) -> Tuple[torch.Tensor, object]:
        """
        Override the original sample method to use regression guidance.
        
        This method automatically uses regression guidance if it's configured,
        otherwise falls back to the original sampling method.
        """
        if self.regression_guidance is not None:
            print("Using regression guidance for sampling...")
            return self.sample_with_regression_guidance(
                batch=batch,
                seed=seed,
                start_from_noisy_image=start_from_noisy_image,
                bf16=bf16,
            )
        else:
            print("No regression guidance configured, using standard sampling...")
            return super().sample(
                batch=batch,
                seed=seed,
                start_from_noisy_image=start_from_noisy_image,
                guidance_pixels=guidance_pixels,
                guidance_scale=guidance_scale,
                bf16=bf16,
            )


def load_custom_model_with_regression_guidance(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    sigma_min: float = 0.02,
    sigma_max: float = 200.0,
    num_steps: int = 18,
    allow_second_order_derivatives: bool = False,
    **kwargs
) -> RegressionGuidedCBottle3d:
    """
    Load a custom model with regression guidance support.
    
    This is a wrapper around the original load_custom_model that returns
    a RegressionGuidedCBottle3d instance instead of CBottle3d.
    
    Args:
        checkpoint_path: Path to your custom checkpoint file (.checkpoint)
        model_name: Optional name for the model (for logging)
        sigma_min: Minimum noise sigma for diffusion sampling
        sigma_max: Maximum noise sigma for diffusion sampling  
        num_steps: Number of sampling steps
        allow_second_order_derivatives: Whether to allow second order derivatives
        **kwargs: Additional arguments passed to CBottle3d
        
    Returns:
        RegressionGuidedCBottle3d: Loaded model ready for regression-guided inference
    """
    from custom_loader import load_custom_model
    
    # Load the base model
    base_model = load_custom_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        allow_second_order_derivatives=allow_second_order_derivatives,
        **kwargs
    )
    
    # Create regression-guided version
    regression_model = RegressionGuidedCBottle3d(
        net=base_model.net,
        batch_info=base_model.batch_info,
        coords=base_model.coords,
        sigma_min=base_model.sigma_min,
        sigma_max=base_model.sigma_max,
        num_steps=base_model.num_steps,
        separate_classifier=base_model.separate_classifier,
        classifier_grid=base_model.classifier_grid,
        icon_mask=base_model.icon_mask,
        era5_mask=base_model.era5_mask,
    )
    
    print(f"Loaded custom model with regression guidance support: {model_name or 'unnamed'}")
    return regression_model


# Example usage
def example_usage():
    """
    Example of how to use regression guidance with a custom model.
    """
    print("Regression-Guided cBottle_tio Example")
    print("=" * 50)
    
    # Load your custom model with regression guidance support
    # model = load_custom_model_with_regression_guidance(
    #     checkpoint_path="/path/to/your/model.checkpoint",
    #     model_name="my-weather-model"
    # )
    
    # Create dummy observation data (replace with real ERA5 data)
    num_obs = 100
    observation_data = torch.randn(num_obs, 2) * 10 + 273.15  # T850, T500 in Kelvin
    observation_locations = torch.randint(0, 1000, (num_obs,))  # Random pixel locations
    observation_variables = ['T850', 'T500']
    
    # Set up regression guidance
    # model.set_regression_guidance(
    #     observation_data=observation_data,
    #     observation_locations=observation_locations,
    #     observation_variables=observation_variables,
    #     observation_uncertainty=0.1,
    #     guidance_scale=1.0,
    # )
    
    # Create dummy batch (replace with your actual data)
    # batch = {
    #     "target": torch.randn(1, 50, 1, 1000),  # [batch, channels, time, pixels]
    #     "labels": torch.zeros(1, 2),
    #     "condition": torch.randn(1, 0, 1, 1000),
    #     "second_of_day": torch.randint(0, 86400, (1,)),
    #     "day_of_year": torch.randint(1, 366, (1,))
    # }
    
    # Sample with regression guidance
    # output, coords = model.sample(batch)
    
    print("Example setup complete!")
    print("To use with your model:")
    print("1. Load your model with load_custom_model_with_regression_guidance()")
    print("2. Set up regression guidance with set_regression_guidance()")
    print("3. Call model.sample() with your batch data")


if __name__ == "__main__":
    example_usage()
