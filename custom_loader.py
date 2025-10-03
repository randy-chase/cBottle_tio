#!/usr/bin/env python3
"""
Custom loader for cBottle models.

This module provides utilities for loading custom trained cBottle models
that aren't included in the default model registry.
"""

import os
import logging
from typing import Optional, Union, List
import cbottle.inference
import cbottle.checkpointing
import cbottle.config.environment


def load_custom_model(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    sigma_min: float = 0.02,
    sigma_max: float = 200.0,
    num_steps: int = 18,
    separate_classifier_path: Optional[str] = None,
    allow_second_order_derivatives: bool = False,
    **kwargs
) -> cbottle.inference.CBottle3d:
    """
    Load a custom cBottle model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to your custom checkpoint file (.checkpoint)
        model_name: Optional name for the model (for logging)
        sigma_min: Minimum noise sigma for diffusion sampling
        sigma_max: Maximum noise sigma for diffusion sampling  
        num_steps: Number of sampling steps
        separate_classifier_path: Optional path to separate classifier for guidance
        allow_second_order_derivatives: Whether to allow second order derivatives
        **kwargs: Additional arguments passed to CBottle3d
        
    Returns:
        CBottle3d: Loaded model ready for inference
        
    Example:
        >>> model = load_custom_model("/path/to/my-model.checkpoint")
        >>> output, coords = model.sample(batch)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    logging.info(f"Loading custom model from: {checkpoint_path}")
    if model_name:
        logging.info(f"Model name: {model_name}")
    
    # Load the model using the checkpointing system
    with cbottle.checkpointing.Checkpoint(checkpoint_path) as c:
        # Read the model configuration and create the model
        model = c.read_model(
            map_location=None,
            allow_second_order_derivatives=allow_second_order_derivatives
        ).cuda().eval()
        
        # Read the batch info (contains channel information)
        batch_info = c.read_batch_info()
        
        # Attach batch_info to the model so CBottle3d can access it
        model.batch_info = batch_info
        logging.info(f"Attached batch_info with {len(batch_info.channels)} channels: {batch_info.channels}")
    
    # Load separate classifier if provided
    separate_classifier = None
    if separate_classifier_path:
        if not os.path.exists(separate_classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {separate_classifier_path}")
        
        logging.info(f"Loading separate classifier from: {separate_classifier_path}")
        with cbottle.checkpointing.Checkpoint(separate_classifier_path) as c:
            separate_classifier = (
                c.read_model(
                    map_location=None,
                    allow_second_order_derivatives=allow_second_order_derivatives
                )
                .cuda()
                .eval()
            )
    
    # Verify that the model has batch_info before creating CBottle3d
    if not hasattr(model, 'batch_info') or model.batch_info is None:
        raise AttributeError("Model does not have batch_info attribute. This is required for CBottle3d.")
    
    # Create the CBottle3d wrapper
    return cbottle.inference.CBottle3d(
        net=model,
        separate_classifier=separate_classifier,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        **kwargs
    )


def load_custom_moe_model(
    checkpoint_paths: List[str],
    sigma_thresholds: List[float],
    model_name: Optional[str] = None,
    separate_classifier_path: Optional[str] = None,
    allow_second_order_derivatives: bool = False,
    **kwargs
) -> cbottle.inference.CBottle3d:
    """
    Load a custom Mixture of Experts (MoE) cBottle model.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files for different experts
        sigma_thresholds: List of sigma thresholds for expert selection
        model_name: Optional name for the model (for logging)
        separate_classifier_path: Optional path to separate classifier for guidance
        allow_second_order_derivatives: Whether to allow second order derivatives
        **kwargs: Additional arguments passed to CBottle3d
        
    Returns:
        CBottle3d: Loaded MoE model ready for inference
        
    Example:
        >>> paths = ["expert1.checkpoint", "expert2.checkpoint", "expert3.checkpoint"]
        >>> thresholds = [100.0, 10.0]
        >>> model = load_custom_moe_model(paths, thresholds)
    """
    # Validate inputs
    if len(checkpoint_paths) < len(sigma_thresholds) + 1:
        raise ValueError("Number of checkpoint paths must be >= number of sigma thresholds + 1")
    
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    logging.info(f"Loading custom MoE model with {len(checkpoint_paths)} experts")
    if model_name:
        logging.info(f"Model name: {model_name}")
    
    # Load the MoE model
    moe_denoiser = cbottle.inference.MixtureOfExpertsDenoiser.from_pretrained(
        checkpoint_paths,
        sigma_thresholds=tuple(sigma_thresholds),
        allow_second_order_derivatives=allow_second_order_derivatives
    )
    
    # The MoE model should already have batch_info from from_pretrained
    # but let's verify it exists
    if not hasattr(moe_denoiser, 'batch_info') or moe_denoiser.batch_info is None:
        # Fallback: load batch_info from the first checkpoint
        with cbottle.checkpointing.Checkpoint(checkpoint_paths[0]) as c:
            batch_info = c.read_batch_info()
            moe_denoiser.batch_info = batch_info
    
    # Load separate classifier if provided
    separate_classifier = None
    if separate_classifier_path:
        if not os.path.exists(separate_classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {separate_classifier_path}")
        
        logging.info(f"Loading separate classifier from: {separate_classifier_path}")
        with cbottle.checkpointing.Checkpoint(separate_classifier_path) as c:
            separate_classifier = (
                c.read_model(
                    map_location=None,
                    allow_second_order_derivatives=allow_second_order_derivatives
                )
                .cuda()
                .eval()
            )
    
    # Verify that the MoE model has batch_info before creating CBottle3d
    if not hasattr(moe_denoiser, 'batch_info') or moe_denoiser.batch_info is None:
        raise AttributeError("MoE model does not have batch_info attribute. This is required for CBottle3d.")
    
    # Create the CBottle3d wrapper
    return cbottle.inference.CBottle3d(
        net=moe_denoiser,
        separate_classifier=separate_classifier,
        **kwargs
    )


def inspect_checkpoint(checkpoint_path: str) -> dict:
    """
    Inspect a checkpoint file to see its contents and configuration.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        dict: Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with cbottle.checkpointing.Checkpoint(checkpoint_path) as c:
        try:
            model_config = c.read_model_config()
            batch_info = c.read_batch_info()
            
            return {
                "model_config": {
                    "architecture": model_config.architecture,
                    "model_channels": model_config.model_channels,
                    "out_channels": model_config.out_channels,
                    "condition_channels": model_config.condition_channels,
                    "time_length": model_config.time_length,
                    "level": model_config.level,
                    "enable_classifier": model_config.enable_classifier,
                },
                "batch_info": {
                    "channels": batch_info.channels,
                    "time_step": batch_info.time_step,
                    "time_unit": batch_info.time_unit.name,
                    "scales": batch_info.scales,
                    "center": batch_info.center,
                }
            }
        except Exception as e:
            return {"error": f"Failed to read checkpoint: {e}"}


# Example usage and testing
if __name__ == "__main__":
    # Example: Load a custom model
    # model = load_custom_model("/path/to/your/model.checkpoint")
    
    # Example: Inspect a checkpoint
    # info = inspect_checkpoint("/path/to/your/model.checkpoint")
    # print("Model channels:", info["batch_info"]["channels"])
    
    print("Custom loader module loaded successfully!")
    print("Available functions:")
    print("- load_custom_model()")
    print("- load_custom_moe_model()") 
    print("- inspect_checkpoint()")
