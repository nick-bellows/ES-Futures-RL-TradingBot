"""
Simple test of PPO inference without Unicode characters
"""

import numpy as np
from ppo_inference import PPOInference, FeatureCalculator

def test_model():
    """Test model inference"""
    print("Testing PPO Model Inference")
    print("=" * 50)
    
    # Load model
    model = PPOInference("ppo_weights.json")
    if not model.load_weights():
        print("FAILED: Could not load weights")
        return
    
    print("SUCCESS: Model loaded")
    
    # Create features
    features = FeatureCalculator(60)
    
    # Test with sample data
    for i in range(20):
        price = 5000 + np.random.normal(0, 5)
        feature_vec = features.calculate_features(price, 1000)
        
        # Test stochastic
        action_s, conf_s = model.predict(feature_vec, stochastic=True)
        
        # Test deterministic  
        action_d, conf_d = model.predict(feature_vec, stochastic=False)
        
        if i < 5:
            actions = ['HOLD', 'BUY', 'SELL']
            print(f"Test {i+1}: Stoch={actions[action_s]} Det={actions[action_d]}")
    
    # Get statistics
    stats = model.get_statistics()
    print(f"\nTotal predictions: {stats['total_predictions']}")
    
    # Check action distribution
    dist = stats['action_distribution']
    print(f"HOLD: {dist['HOLD']:.1%}")
    print(f"BUY:  {dist['BUY']:.1%}")
    print(f"SELL: {dist['SELL']:.1%}")
    
    trading_pct = dist['BUY'] + dist['SELL']
    print(f"Trading activity: {trading_pct:.1%}")
    
    if trading_pct > 0.1:
        print("SUCCESS: Model shows trading activity")
    else:
        print("WARNING: Low trading activity")
    
    print("Test complete!")

if __name__ == "__main__":
    test_model()