# Package Installation Guide for Regression Guidance

## 🎯 **Problem Solved**

You asked: *"why can't the scripts just be found after I install the repo?"*

The answer is that the regression guidance modules needed to be properly integrated into the cBottle_tio package structure. I've now fixed this by:

1. **Moving modules into the proper package structure** (`src/cbottle/`)
2. **Updating the package's `__init__.py`** to expose the new modules
3. **Creating proper import paths** that work after installation

## 📦 **Package Structure**

The regression guidance modules are now properly integrated into the cBottle_tio package:

```
src/cbottle/
├── __init__.py                          # Updated to expose regression guidance
├── regression_guidance.py               # Core regression guidance system
├── regression_guided_inference.py       # Extended CBottle3d with regression guidance
├── amip_regression_utils.py             # AMIP dataset utilities
└── ... (existing cBottle modules)
```

## 🚀 **How to Use After Installation**

### Install cBottle_tio as a Package

```bash
cd /path/to/cBottle_tio
pip install -e .
```

### Use the Regression Guidance

```python
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
```

## 🔧 **Available Functions**

After installation, you can import these functions directly from the `cbottle` package:

```python
import cbottle

# Main functions
cbottle.quick_regression_guidance_setup()           # One-line setup
cbottle.load_custom_model_with_regression_guidance() # Load model with regression support
cbottle.setup_regression_guidance_with_amip()       # Set up guidance with AMIP data

# Classes
cbottle.RegressionGuidedCBottle3d                   # Extended CBottle3d class
cbottle.RegressionGuidance                          # Core guidance system
```

## 📋 **Installation Steps**

1. **Navigate to cBottle_tio directory**:
   ```bash
   cd /Users/randychase/Documents/PythonWorkspace/cbottle/cBottle_tio
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```python
   import cbottle
   print(dir(cbottle))  # Should show the new regression guidance functions
   ```

4. **Test the regression guidance**:
   ```python
   import cbottle
   
   # Test imports
   from cbottle import RegressionGuidance, quick_regression_guidance_setup
   print("✅ All imports successful!")
   ```

## 🧪 **Test Script**

Run the test script to verify everything works:

```bash
python examples/regression_guidance_package_example.py
```

This will test:
- Package imports
- Module functionality
- AMIP dataset integration

## 📚 **Updated Examples**

All examples now use the proper package imports:

- **`examples/regression_guidance_package_example.py`** - Uses `import cbottle`
- **`examples/regression_guidance_with_amip.py`** - Complete AMIP integration
- **`examples/regression_guidance_example.py`** - Basic usage

## 🎉 **Benefits of Package Installation**

1. **Clean imports**: `import cbottle` instead of complex path manipulation
2. **Proper module discovery**: Python can find the modules automatically
3. **Version management**: Proper package versioning and dependencies
4. **IDE support**: Better autocomplete and type checking
5. **Distribution**: Easy to share and install on other systems

## 🔄 **Migration from Manual Imports**

If you were using the manual import approach before:

**Old way (manual imports)**:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regression_guided_inference import load_custom_model_with_regression_guidance
```

**New way (package imports)**:
```python
import cbottle
model = cbottle.load_custom_model_with_regression_guidance(...)
```

## ✅ **Verification**

After installation, you should be able to run:

```python
import cbottle
print("Available regression guidance functions:")
print([attr for attr in dir(cbottle) if 'regression' in attr.lower() or 'guidance' in attr.lower()])
```

This should show:
- `RegressionGuidedCBottle3d`
- `load_custom_model_with_regression_guidance`
- `RegressionGuidance`
- `quick_regression_guidance_setup`
- `setup_regression_guidance_with_amip`

## 🎯 **Summary**

The regression guidance system is now properly integrated into the cBottle_tio package. After installing the package with `pip install -e .`, you can use all the regression guidance functionality with clean, simple imports like `import cbottle`.

No more import errors or path manipulation needed! 🚀
