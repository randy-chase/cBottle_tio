#!/usr/bin/env python3
"""
Test script for regression guidance with cBottle_tio.

This script tests the regression guidance implementation without requiring
a real model checkpoint, using synthetic data and the AMIP dataset.
"""

import os
import sys
import torch
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_regression_guidance_components():
    """Test the core regression guidance components."""
    print("Testing regression guidance components...")
    
    try:
        from regression_guidance import RegressionGuidance, create_observation_data_from_era5
        
        # Create dummy data
        num_obs = 50
        num_vars = 3
        observation_data = torch.randn(num_obs, num_vars) * 10 + 273.15
        observation_locations = torch.randint(0, 1000, (num_obs,))
        observation_variables = ['T850', 'T500', 'T300']
        
        # Create mock batch info
        class MockBatchInfo:
            def __init__(self):
                self.channels = [
                    'U1000', 'U850', 'U700', 'U500', 'U300', 'U200', 'U50', 'U10',
                    'V1000', 'V850', 'V700', 'V500', 'V300', 'V200', 'V50', 'V10', 
                    'T1000', 'T850', 'T700', 'T500', 'T300', 'T200', 'T50', 'T10',
                    'Z1000', 'Z850', 'Z700', 'Z500', 'Z300', 'Z200', 'Z50', 'Z10',
                    'Q1000', 'Q850', 'Q700', 'Q500', 'Q300', 'Q200', 'Q50', 'Q10',
                    'tcwv', 'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut', 
                    'pres_msl', 'pr', 'rsds', 'sst', 'sic'
                ]
        
        batch_info = MockBatchInfo()
        
        # Test RegressionGuidance
        guidance = RegressionGuidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=0.1,
            guidance_scale=1.0,
            batch_info=batch_info
        )
        
        print("✅ RegressionGuidance created successfully")
        
        # Test forward operator creation
        model_output_shape = (1, len(batch_info.channels), 1, 1000)
        forward_operator = guidance.create_forward_operator(model_output_shape)
        
        # Test forward operator with dummy data
        dummy_output = torch.randn(*model_output_shape)
        extracted = forward_operator(dummy_output)
        
        print(f"✅ Forward operator works: input {dummy_output.shape} -> output {extracted.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing regression guidance components: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_amip_utils():
    """Test the AMIP utility functions."""
    print("\nTesting AMIP utility functions...")
    
    try:
        from amip_regression_utils import create_synthetic_observations, create_observation_locations
        
        # Test synthetic observations
        observation_data, observation_locations = create_synthetic_observations(
            observation_variables=['T850', 'T500', 'T300'],
            num_obs=50
        )
        
        print(f"✅ Synthetic observations created: {observation_data.shape}")
        
        # Test observation locations creation
        class MockBatch:
            def __init__(self):
                self.target = torch.randn(1, 50, 1, 1000)
        
        mock_batch = MockBatch()
        obs_locs = create_observation_locations(mock_batch, num_obs=50, strategy="random")
        
        print(f"✅ Observation locations created: {obs_locs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AMIP utils: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_amip_dataset():
    """Test loading the AMIP dataset."""
    print("\nTesting AMIP dataset loading...")
    
    try:
        from cbottle.datasets.dataset_3d import get_dataset
        
        # Try to load AMIP dataset
        ds = get_dataset(dataset="amip")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        
        print(f"✅ AMIP dataset loaded successfully")
        print(f"  - Target shape: {batch['target'].shape}")
        print(f"  - Condition shape: {batch['condition'].shape}")
        print(f"  - Labels shape: {batch['labels'].shape}")
        print(f"  - Time info: {batch['second_of_day']} seconds, day {batch['day_of_year']}")
        
        # Check if we have the expected variables
        if hasattr(ds, 'batch_info'):
            print(f"  - Available variables: {len(ds.batch_info.channels)}")
            temp_vars = [var for var in ds.batch_info.channels if var.startswith('T')]
            print(f"  - Temperature variables: {temp_vars}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading AMIP dataset: {e}")
        print("This might be expected if AMIP data is not available")
        return False


def test_integration():
    """Test the full integration."""
    print("\nTesting full integration...")
    
    try:
        # Test if we can import all the main modules
        from regression_guided_inference import RegressionGuidedCBottle3d, load_custom_model_with_regression_guidance
        from regression_guidance import RegressionGuidance
        from amip_regression_utils import quick_regression_guidance_setup
        
        print("✅ All modules imported successfully")
        
        # Test creating a mock model (without loading a real checkpoint)
        class MockModel:
            def __init__(self):
                self.batch_info = type('BatchInfo', (), {
                    'channels': ['T850', 'T500', 'T300', 'U850', 'V850']
                })()
                self.regression_guidance = None
            
            def set_regression_guidance(self, **kwargs):
                self.regression_guidance = RegressionGuidance(**kwargs)
                print("✅ Mock model regression guidance set")
        
        mock_model = MockModel()
        
        # Test setting up regression guidance
        observation_data = torch.randn(50, 3) * 10 + 273.15
        observation_locations = torch.randint(0, 1000, (50,))
        observation_variables = ['T850', 'T500', 'T300']
        
        mock_model.set_regression_guidance(
            observation_data=observation_data,
            observation_locations=observation_locations,
            observation_variables=observation_variables,
            observation_uncertainty=0.1,
            guidance_scale=1.0,
            batch_info=mock_model.batch_info
        )
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Regression Guidance Test Suite")
    print("=" * 40)
    
    tests = [
        ("Regression Guidance Components", test_regression_guidance_components),
        ("AMIP Utility Functions", test_amip_utils),
        ("AMIP Dataset Loading", test_amip_dataset),
        ("Full Integration", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! The regression guidance system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
