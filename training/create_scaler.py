#!/usr/bin/env python3
"""
Create Feature Scaler
Creates a basic StandardScaler for the 47 features if one doesn't exist.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tradovate_integration import FeatureCalculator, BarData
from datetime import datetime, timedelta

def create_default_scaler():
    """Create a basic scaler with reasonable default parameters for 47 features"""
    print("Creating default feature scaler...")
    
    # Generate sample data to fit the scaler
    calculator = FeatureCalculator(lookback_bars=60)
    
    # Add enough sample bars
    base_time = datetime.now()
    base_price = 4500.0
    
    print("Generating sample data...")
    for i in range(65):
        bar = BarData(
            timestamp=base_time + timedelta(minutes=i),
            open=base_price + np.random.randn() * 2.0,
            high=base_price + np.random.randn() * 2.0 + 1.0,
            low=base_price + np.random.randn() * 2.0 - 1.0,
            close=base_price + np.random.randn() * 2.0,
            volume=1000 + int(np.random.randn() * 200)
        )
        calculator.add_bar(bar)
    
    # Generate multiple feature sets for better scaling
    feature_samples = []
    
    print("Collecting feature samples...")
    for i in range(100):  # Generate 100 samples
        # Add one more bar to trigger recalculation
        bar = BarData(
            timestamp=base_time + timedelta(minutes=65 + i),
            open=base_price + np.random.randn() * 5.0,
            high=base_price + np.random.randn() * 5.0 + 2.0,
            low=base_price + np.random.randn() * 5.0 - 2.0,
            close=base_price + np.random.randn() * 5.0,
            volume=1000 + int(np.random.randn() * 500)
        )
        calculator.add_bar(bar)
        
        if calculator.is_ready():
            features = calculator.calculate_features()
            if features is not None and len(features) == 47:
                feature_samples.append(features)
    
    if len(feature_samples) < 10:
        print("Warning: Could not generate enough feature samples, using simple defaults")
        # Create scaler with default mean=0, std=1 for 47 features
        scaler = StandardScaler()
        # Fit with dummy data
        dummy_data = np.random.randn(100, 47)
        scaler.fit(dummy_data)
    else:
        print(f"Generated {len(feature_samples)} feature samples")
        feature_array = np.array(feature_samples)
        
        # Create and fit the scaler
        scaler = StandardScaler()
        scaler.fit(feature_array)
        
        print(f"Scaler statistics:")
        print(f"  Mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
        print(f"  Scale range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")
    
    # Save the scaler
    scaler_path = "models/feature_scaler.pkl"
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Feature scaler saved to: {scaler_path}")
    
    # Test the scaler
    test_features = np.random.randn(47)
    scaled = scaler.transform(test_features.reshape(1, -1))
    print(f"Test scaling: {test_features.shape} -> {scaled.shape}")
    
    return scaler_path

if __name__ == "__main__":
    scaler_path = create_default_scaler()
    print(f"\nScaler created successfully at: {scaler_path}")
    print("The scaler is now ready for use with your model.")