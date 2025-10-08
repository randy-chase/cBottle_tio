# Regression Guidance for cBottle_tio

This implementation adds **Score-Based Data Assimilation** capabilities to cBottle_tio by replacing the classifier-based hurricane guidance with regression guidance using weather observations (e.g., ERA5 data).

## Overview

The regression guidance system constrains the diffusion sampling process to match observed weather data, similar to the approach used in the [appa repository](https://github.com/montefiore-sail/appa) but adapted for cBottle_tio's architecture.

### Key Features

- **Physics-based guidance**: Uses regression loss instead of classifier guidance
- **ERA5 integration**: Works with ERA5 temperature observations at multiple pressure levels
- **EDM sampling compatibility**: Integrates with cBottle_tio's existing EDM sampling
- **Flexible observation setup**: Supports arbitrary observation locations and variables
- **Uncertainty handling**: Includes observation uncertainty in the guidance process

## Files

- `regression_guidance.py`: Core regression guidance implementation
- `regression_guided_inference.py`: Extended CBottle3d class with regression guidance
- `era5_data_loader.py`: Utilities for loading ERA5 data
- `examples/regression_guidance_example.py`: Complete example script

## Quick Start

### Method 1: Using AMIP Dataset (Recommended)

```python
import cbottle

# Quick setup with AMIP dataset
model, batch = cbottle.quick_regression_guidance_setup(
    checkpoint_path="/path/to/your/model.checkpoint",
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100,
    guidance_scale=1.0,
    use_amip=True  # Uses real AMIP data with proper time encoding and SST
)

# Perform data assimilation
output, coords = model.sample(batch, seed=42)
```

### Method 1b: Custom Variables (Fix Channel Mismatch)

If you get a `RuntimeError: The size of tensor a (53) must match the size of tensor b (45)` error, your model has a different variable configuration than the default. Use the `custom_variables` parameter:

```python
import cbottle

# Your custom model's variables
custom_variables = [
    'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
    'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10',
    'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
    'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
    'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
    'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut',
    'pres_msl', 'pr', 'rsds', 'sst', 'sic'
]

# Quick setup with custom variables
model, batch = cbottle.quick_regression_guidance_setup(
    checkpoint_path="/path/to/your/model.checkpoint",
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100,
    guidance_scale=1.0,
    use_amip=True,
    custom_variables=custom_variables  # This fixes the channel mismatch!
)

# Perform data assimilation
output, coords = model.sample(batch, seed=42)
```

### Method 2: Manual Setup with AMIP Dataset

```python
import cbottle
from cbottle.datasets.dataset_3d import get_dataset

# 1. Load your model
model = cbottle.load_custom_model_with_regression_guidance(
    checkpoint_path="/path/to/your/model.checkpoint",
    model_name="my-weather-model"
)

# 2. Load AMIP dataset for proper batch preparation
ds = get_dataset(dataset="amip")
loader = torch.utils.data.DataLoader(ds, batch_size=1)
batch = next(iter(loader))

# 3. Set up regression guidance using AMIP data
cbottle.setup_regression_guidance_with_amip(
    model=model,
    amip_batch=batch,
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100,
    observation_strategy="random",  # or "uniform" or "land_only"
    guidance_scale=1.0
)

# 4. Perform data assimilation
output, coords = model.sample(batch)
```

### Method 3: Manual Setup with Synthetic Data

```python
import cbottle
import torch

# 1. Load your model
model = cbottle.load_custom_model_with_regression_guidance(
    checkpoint_path="/path/to/your/model.checkpoint",
    model_name="my-weather-model"
)

# 2. Create synthetic observation data
observation_data = torch.randn(100, 3) * 10 + 273.15  # T850, T500, T300 in Kelvin
observation_locations = torch.randint(0, 1000, (100,))  # Random pixel locations
observation_variables = ['T850', 'T500', 'T300']

# 3. Set up regression guidance
model.set_regression_guidance(
    observation_data=observation_data,
    observation_locations=observation_locations,
    observation_variables=observation_variables,
    observation_uncertainty=0.1,  # 0.1 K uncertainty
    guidance_scale=1.0,  # Guidance strength
)

# 4. Create batch data
batch = {
    "target": torch.randn(1, 50, 1, 1000),  # [batch, channels, time, pixels]
    "labels": torch.zeros(1, 2),
    "condition": torch.randn(1, 0, 1, 1000),
    "second_of_day": torch.randint(0, 86400, (1,)),
    "day_of_year": torch.randint(1, 366, (1,))
}

# 5. Sample with regression guidance
output, coords = model.sample(batch)
```

## How It Works

### Regression Guidance vs Classifier Guidance

**Original cBottle_tio (Classifier Guidance):**
- Uses a classifier head to predict hurricane probability at each pixel
- Guidance steers sampling toward higher hurricane probabilities
- Binary cross-entropy loss with logits

**New Regression Guidance:**
- Uses observed weather data (e.g., ERA5 temperatures)
- Guidance steers sampling toward observed values
- Mean squared error loss with uncertainty weighting

### Mathematical Formulation

The regression guidance modifies the denoising process by adding a guidance term:

```
d_guided = d_denoised + guidance_scale * ∇_x L_regression(x, y_obs)
```

Where:
- `d_denoised`: Standard denoised prediction
- `L_regression`: Regression loss between predicted and observed values
- `y_obs`: Observed weather data
- `guidance_scale`: Strength of the guidance

The regression loss is:
```
L_regression = ||A(x) - y_obs||² / (2σ²)
```

Where:
- `A(x)`: Forward operator that extracts observed variables from model output
- `σ`: Observation uncertainty

## AMIP Dataset Integration

### Using AMIP Dataset (Recommended)

The AMIP dataset provides real atmospheric data with proper time encoding, SST conditions, and all the variables your model expects. This is the recommended approach for regression guidance.

```python
from cbottle.datasets.dataset_3d import get_dataset
from amip_regression_utils import setup_regression_guidance_with_amip

# Load AMIP dataset
ds = get_dataset(dataset="amip")
loader = torch.utils.data.DataLoader(ds, batch_size=1)
batch = next(iter(loader))

# Set up regression guidance using AMIP data
setup_regression_guidance_with_amip(
    model=model,
    amip_batch=batch,
    observation_variables=['T850', 'T500', 'T300'],
    num_obs=100,
    observation_strategy="random"  # or "uniform" or "land_only"
)
```

### AMIP Batch Structure

The AMIP dataset provides batches with the following structure:

```python
batch = {
    "target": torch.Tensor,      # [batch, channels, time, pixels] - Atmospheric variables
    "condition": torch.Tensor,   # [batch, condition_channels, time, pixels] - SST data
    "labels": torch.Tensor,      # [batch, num_classes] - Dataset labels
    "second_of_day": torch.Tensor,  # [batch] - Time of day in seconds
    "day_of_year": torch.Tensor,    # [batch] - Day of year
    "timestamp": cftime.DatetimeGregorian  # Timestamp object
}
```

### Observation Strategies

- **"random"**: Randomly select observation locations
- **"uniform"**: Uniformly distributed observations across the globe
- **"land_only"**: Select only land pixels (where SST is NaN)

## ERA5 Data Integration

### Loading ERA5 Data from NetCDF Files

```python
from era5_data_loader import load_era5_for_regression_guidance

# Load ERA5 data for regression guidance
observation_data, observation_locations = load_era5_for_regression_guidance(
    era5_file_path="/path/to/era5_data.nc",
    observation_variables=['T850', 'T500', 'T300'],
    batch_info=model.batch_info,
    num_pixels=1000
)
```

### Variable Mapping

The system maps cBottle variable names to ERA5 variable names:

| cBottle Variable | ERA5 Variable | Description |
|------------------|---------------|-------------|
| T850 | t | Temperature at 850 hPa |
| T500 | t | Temperature at 500 hPa |
| T300 | t | Temperature at 300 hPa |
| U850 | u | U-wind at 850 hPa |
| V850 | v | V-wind at 850 hPa |
| ... | ... | ... |

## Configuration Parameters

### RegressionGuidance Parameters

- `observation_data`: Observed values [num_obs, num_variables]
- `observation_locations`: Pixel indices [num_obs] where observations are available
- `observation_variables`: List of variable names being observed
- `observation_uncertainty`: Standard deviation of observation errors (default: 0.1)
- `guidance_scale`: Strength of guidance (default: 1.0)

### Tuning Guidelines

- **observation_uncertainty**: 
  - Lower values = stronger constraint (more trust in observations)
  - Higher values = weaker constraint (less trust in observations)
  - Typical range: 0.05 - 0.5 K for temperature

- **guidance_scale**:
  - Lower values = weaker guidance effect
  - Higher values = stronger guidance effect
  - Typical range: 0.1 - 2.0

## Example: Complete Data Assimilation Pipeline

```python
#!/usr/bin/env python3
import torch
from regression_guided_inference import load_custom_model_with_regression_guidance
from era5_data_loader import load_era5_for_regression_guidance

# 1. Load model
model = load_custom_model_with_regression_guidance(
    checkpoint_path="/path/to/your/model.checkpoint",
    model_name="weather-model"
)

# 2. Load ERA5 observations
observation_data, observation_locations = load_era5_for_regression_guidance(
    era5_file_path="/path/to/era5_data.nc",
    observation_variables=['T850', 'T500', 'T300'],
    batch_info=model.batch_info
)

# 3. Set up regression guidance
model.set_regression_guidance(
    observation_data=observation_data,
    observation_locations=observation_locations,
    observation_variables=['T850', 'T500', 'T300'],
    observation_uncertainty=0.1,
    guidance_scale=1.0
)

# 4. Create batch
batch = {
    "target": torch.randn(1, 50, 1, 1000),
    "labels": torch.zeros(1, 2),
    "condition": torch.randn(1, 0, 1, 1000),
    "second_of_day": torch.randint(0, 86400, (1,)),
    "day_of_year": torch.randint(1, 366, (1,))
}

# 5. Perform data assimilation
output, coords = model.sample(batch, seed=42)

print(f"Data assimilation complete! Output shape: {output.shape}")
```

## Comparison with Appa

| Feature | Appa (MMPS) | cBottle_tio (Regression Guidance) |
|---------|-------------|-----------------------------------|
| **Approach** | MMPS with iterative solver | Direct regression guidance |
| **Complexity** | High (GMRES solver) | Medium (gradient-based) |
| **Performance** | Slower (iterative) | Faster (direct) |
| **Accuracy** | Very high | High |
| **Implementation** | Complex | Simpler |

## Troubleshooting

### Common Issues

1. **Channel Mismatch Error**: `RuntimeError: The size of tensor a (53) must match the size of tensor b (45)`
   - **Cause**: Your custom model has a different variable configuration than the default
   - **Solution**: Use the `custom_variables` parameter in `quick_regression_guidance_setup()`:
   ```python
   custom_variables = [
       'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
       'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10',
       'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
       'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
       'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
       'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut',
       'pres_msl', 'pr', 'rsds', 'sst', 'sic'
   ]
   
   model, batch = cbottle.quick_regression_guidance_setup(
       checkpoint_path="/path/to/your/model.checkpoint",
       custom_variables=custom_variables  # This fixes the mismatch!
   )
   ```

2. **Batch Info Property Error**: `AttributeError: property 'batch_info' of 'RegressionGuidedCBottle3d' object has no setter`
   - **Cause**: The `batch_info` property is read-only and cannot be modified after model creation
   - **Solution**: This is now automatically handled by the `custom_variables` parameter. The fix sets the correct `batch_info` during model loading instead of trying to modify it afterward.

3. **Condition Tensor Shape Error**: `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 0 but got size 1 for tensor number 1 in the list.`
   - **Cause**: The condition tensor has the wrong number of channels for the model's configuration
   - **Solution**: The fix automatically detects the model's `condition_channels` from the checkpoint and creates the correct condition tensor shape.

4. **"Regression guidance not set"**
   - Make sure to call `model.set_regression_guidance()` before sampling

5. **"Variable not found in batch_info"**
   - Check that your observation variables match the model's variable names
   - Use `model.batch_info.channels` to see available variables

6. **Shape mismatches**
   - Ensure observation data shape matches [num_obs, num_variables]
   - Check that observation_locations are valid pixel indices

7. **Memory issues**
   - Reduce the number of observations
   - Use smaller guidance_scale values
   - Enable gradient checkpointing

### Performance Tips

- Use `bf16=True` for faster sampling
- Reduce `num_steps` for faster but lower quality results
- Tune `guidance_scale` for optimal balance between constraint and quality
- Use GPU acceleration for large observation datasets

## Future Enhancements

- [ ] Support for multiple observation types (stations, satellites, radar)
- [ ] Adaptive guidance scaling based on observation quality
- [ ] Integration with more sophisticated data assimilation methods
- [ ] Support for temporal observations
- [ ] Uncertainty quantification in the output

## References

- [Appa: Score-Based Data Assimilation](https://github.com/montefiore-sail/appa)
- [cBottle_tio: Hurricane Guidance](https://github.com/NVIDIA/cBottle_tio)
- [Score-Based Data Assimilation (Rozet et al., 2023)](https://arxiv.org/abs/2306.10574)
- [Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)](https://arxiv.org/abs/2405.13712)
