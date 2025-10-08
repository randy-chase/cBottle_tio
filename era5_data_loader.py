#!/usr/bin/env python3
"""
Utility for loading ERA5 data for regression guidance.

This module provides functions to load ERA5 data and convert it to the format
needed for regression guidance in cBottle_tio.
"""

import torch
import numpy as np
import xarray as xr
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path


def load_era5_from_netcdf(
    file_path: str,
    variables: List[str],
    time_slice: Optional[slice] = None,
    lat_slice: Optional[slice] = None,
    lon_slice: Optional[slice] = None,
) -> xr.Dataset:
    """
    Load ERA5 data from NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        variables: List of variable names to load
        time_slice: Time slice to extract (optional)
        lat_slice: Latitude slice to extract (optional)
        lon_slice: Longitude slice to extract (optional)
        
    Returns:
        xarray Dataset with ERA5 data
    """
    print(f"Loading ERA5 data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5 file not found: {file_path}")
    
    # Load the dataset
    ds = xr.open_dataset(file_path)
    
    # Select variables
    if variables:
        available_vars = [var for var in variables if var in ds.data_vars]
        missing_vars = [var for var in variables if var not in ds.data_vars]
        
        if missing_vars:
            print(f"Warning: Variables not found in file: {missing_vars}")
        
        if available_vars:
            ds = ds[available_vars]
        else:
            raise ValueError(f"None of the requested variables found in file: {variables}")
    
    # Apply slices if provided
    if time_slice is not None:
        ds = ds.isel(time=time_slice)
    if lat_slice is not None:
        ds = ds.isel(latitude=lat_slice)
    if lon_slice is not None:
        ds = ds.isel(longitude=lon_slice)
    
    print(f"Loaded ERA5 data with shape: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    
    return ds


def convert_era5_to_tensor(
    ds: xr.Dataset,
    variable_mapping: Dict[str, str],
    target_shape: Tuple[int, int, int],  # (channels, time, pixels)
) -> torch.Tensor:
    """
    Convert ERA5 xarray Dataset to PyTorch tensor.
    
    Args:
        ds: xarray Dataset with ERA5 data
        variable_mapping: Mapping from cBottle variable names to ERA5 variable names
        target_shape: Target tensor shape (channels, time, pixels)
        
    Returns:
        PyTorch tensor with ERA5 data
    """
    num_channels, time_length, num_pixels = target_shape
    
    # Initialize output tensor
    era5_tensor = torch.zeros(num_channels, time_length, num_pixels)
    
    # Get coordinate information
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lats = ds.latitude.values
        lons = ds.longitude.values
        print(f"ERA5 grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} points")
    else:
        print("Warning: No latitude/longitude coordinates found")
    
    # Convert each variable
    for cBottle_var, era5_var in variable_mapping.items():
        if era5_var in ds.data_vars:
            # Get the data
            data = ds[era5_var].values
            
            # Handle different data shapes
            if data.ndim == 3:  # (time, lat, lon)
                data = data.reshape(data.shape[0], -1)  # Flatten spatial dimensions
            elif data.ndim == 4:  # (time, level, lat, lon)
                # For pressure level data, you might want to select specific levels
                # For now, take the first level
                data = data[:, 0, :, :].reshape(data.shape[0], -1)
            elif data.ndim == 2:  # (lat, lon)
                data = data.reshape(1, -1)  # Add time dimension
            else:
                print(f"Warning: Unexpected data shape for {era5_var}: {data.shape}")
                continue
            
            # Find the channel index for this variable
            # This assumes you have a way to map variable names to channel indices
            # You'll need to implement this based on your model's variable ordering
            channel_idx = get_channel_index(cBottle_var)
            
            if channel_idx is not None and channel_idx < num_channels:
                # Ensure time dimension matches
                if data.shape[0] > time_length:
                    data = data[:time_length, :]
                elif data.shape[0] < time_length:
                    # Pad with the last available time step
                    padding = np.repeat(data[-1:, :], time_length - data.shape[0], axis=0)
                    data = np.concatenate([data, padding], axis=0)
                
                # Ensure spatial dimension matches
                if data.shape[1] > num_pixels:
                    data = data[:, :num_pixels]
                elif data.shape[1] < num_pixels:
                    # Pad with zeros or repeat
                    padding = np.zeros((data.shape[0], num_pixels - data.shape[1]))
                    data = np.concatenate([data, padding], axis=1)
                
                # Convert to tensor and store
                era5_tensor[channel_idx, :, :] = torch.from_numpy(data.astype(np.float32))
                print(f"Converted {era5_var} -> {cBottle_var} (channel {channel_idx})")
            else:
                print(f"Warning: Could not map {cBottle_var} to channel index")
    
    return era5_tensor


def get_channel_index(variable_name: str) -> Optional[int]:
    """
    Get the channel index for a given variable name.
    
    This function needs to be customized based on your model's variable ordering.
    """
    # Your model's variable keys in order
    variable_keys = [
        'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
        'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10', 
        'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
        'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
        'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
        'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut', 
        'pres_msl', 'pr', 'rsds', 'sst', 'sic'
    ]
    
    try:
        return variable_keys.index(variable_name)
    except ValueError:
        return None


def create_era5_variable_mapping() -> Dict[str, str]:
    """
    Create mapping from cBottle variable names to ERA5 variable names.
    
    Returns:
        Dictionary mapping cBottle variable names to ERA5 variable names
    """
    mapping = {
        # Temperature
        'T1000': 't',  # Surface temperature
        'T850': 't',   # Temperature at 850 hPa
        'T700': 't',   # Temperature at 700 hPa
        'T500': 't',   # Temperature at 500 hPa
        'T300': 't',   # Temperature at 300 hPa
        'T200': 't',   # Temperature at 200 hPa
        'T50': 't',    # Temperature at 50 hPa
        'T10': 't',    # Temperature at 10 hPa
        
        # U-wind
        'U1000': 'u',  # U-wind at 1000 hPa
        'U850': 'u',   # U-wind at 850 hPa
        'U700': 'u',   # U-wind at 700 hPa
        'U500': 'u',   # U-wind at 500 hPa
        'U300': 'u',   # U-wind at 300 hPa
        'U200': 'u',   # U-wind at 200 hPa
        'U50': 'u',    # U-wind at 50 hPa
        'U10': 'u',    # U-wind at 10 hPa
        
        # V-wind
        'V1000': 'v',  # V-wind at 1000 hPa
        'V850': 'v',   # V-wind at 850 hPa
        'V700': 'v',   # V-wind at 700 hPa
        'V500': 'v',   # V-wind at 500 hPa
        'V300': 'v',   # V-wind at 300 hPa
        'V200': 'v',   # V-wind at 200 hPa
        'V50': 'v',    # V-wind at 50 hPa
        'V10': 'v',    # V-wind at 10 hPa
        
        # Geopotential
        'Z1000': 'z',  # Geopotential at 1000 hPa
        'Z850': 'z',   # Geopotential at 850 hPa
        'Z700': 'z',   # Geopotential at 700 hPa
        'Z500': 'z',   # Geopotential at 500 hPa
        'Z300': 'z',   # Geopotential at 300 hPa
        'Z200': 'z',   # Geopotential at 200 hPa
        'Z50': 'z',    # Geopotential at 50 hPa
        'Z10': 'z',    # Geopotential at 10 hPa
        
        # Specific humidity
        'Q1000': 'q',  # Specific humidity at 1000 hPa
        'Q850': 'q',   # Specific humidity at 850 hPa
        'Q700': 'q',   # Specific humidity at 700 hPa
        'Q500': 'q',   # Specific humidity at 500 hPa
        'Q300': 'q',   # Specific humidity at 300 hPa
        'Q200': 'q',   # Specific humidity at 200 hPa
        'Q50': 'q',    # Specific humidity at 50 hPa
        'Q10': 'q',    # Specific humidity at 10 hPa
        
        # Surface variables
        'tas': 't2m',      # 2m temperature
        'uas': 'u10',      # 10m U-wind
        'vas': 'v10',      # 10m V-wind
        'pres_msl': 'msl', # Mean sea level pressure
        'pr': 'tp',        # Total precipitation
        'sst': 'sst',      # Sea surface temperature
        'sic': 'siconc',   # Sea ice concentration
        
        # Radiation
        'rlut': 'ttr',     # Top net thermal radiation
        'rsut': 'str',     # Top net solar radiation
        'rsds': 'ssrd',    # Surface solar radiation downwards
        
        # Other
        'tcwv': 'tcwv',    # Total column water vapour
        'cllvi': 'cllvi',  # Low cloud ice water
        'clivi': 'clivi',  # Ice water path
    }
    
    return mapping


def load_era5_for_regression_guidance(
    era5_file_path: str,
    observation_variables: List[str],
    batch_info,
    num_pixels: int = 1000,
    time_slice: Optional[slice] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load ERA5 data specifically for regression guidance.
    
    Args:
        era5_file_path: Path to ERA5 NetCDF file
        observation_variables: List of variables to observe
        batch_info: cBottle batch info for variable mapping
        num_pixels: Number of pixels in the model grid
        time_slice: Time slice to extract
        
    Returns:
        observation_data: [num_pixels, num_variables] tensor
        observation_locations: [num_pixels] tensor of pixel indices
    """
    print(f"Loading ERA5 data for regression guidance...")
    
    # Load ERA5 data
    ds = load_era5_from_netcdf(
        file_path=era5_file_path,
        variables=observation_variables,
        time_slice=time_slice
    )
    
    # Create variable mapping
    variable_mapping = create_era5_variable_mapping()
    
    # Filter mapping to only include observed variables
    observed_mapping = {var: variable_mapping[var] for var in observation_variables 
                       if var in variable_mapping}
    
    # Convert to tensor
    target_shape = (len(batch_info.channels), 1, num_pixels)
    era5_tensor = convert_era5_to_tensor(ds, observed_mapping, target_shape)
    
    # Extract observation data
    variable_to_channel = {var: i for i, var in enumerate(batch_info.channels)}
    observed_channels = [variable_to_channel[var] for var in observation_variables]
    
    # Create observation data [num_pixels, num_variables]
    observation_data = torch.zeros(num_pixels, len(observation_variables))
    for i, channel_idx in enumerate(observed_channels):
        observation_data[:, i] = era5_tensor[channel_idx, 0, :]  # Take first time step
    
    # Create observation locations (all pixels for now)
    observation_locations = torch.arange(num_pixels)
    
    print(f"Created observation data with shape: {observation_data.shape}")
    print(f"Variables: {observation_variables}")
    
    return observation_data, observation_locations


# Example usage
def example_era5_loading():
    """
    Example of how to load ERA5 data for regression guidance.
    """
    print("ERA5 Data Loading Example")
    print("=" * 30)
    
    # Example file path (update with your actual ERA5 file)
    era5_file_path = "/path/to/your/era5_data.nc"
    
    # Variables to observe
    observation_variables = ['T850', 'T500', 'T300']
    
    # Check if file exists
    if not os.path.exists(era5_file_path):
        print(f"⚠️  ERA5 file not found: {era5_file_path}")
        print("Please update the file path with your actual ERA5 data.")
        return
    
    try:
        # Load ERA5 data
        ds = load_era5_from_netcdf(
            file_path=era5_file_path,
            variables=observation_variables
        )
        
        print("✅ ERA5 data loaded successfully!")
        print(f"Dataset info: {ds}")
        
        # Create variable mapping
        mapping = create_era5_variable_mapping()
        print(f"Variable mapping: {mapping}")
        
    except Exception as e:
        print(f"❌ Error loading ERA5 data: {e}")


if __name__ == "__main__":
    example_era5_loading()
