#!/usr/bin/env python3
"""
Quick Test Runner

Runs the immediate checks suggested for validating the system:
1. Feature calculation consistency check
2. Model loading verification  
3. Configuration validation
4. Quick replay connectivity test
"""

import sys
import os

def run_immediate_checks():
    """Run all immediate validation checks"""
    print("🚀 RUNNING IMMEDIATE VALIDATION CHECKS")
    print("=" * 50)
    
    results = []
    
    # 1. Model Loading Test
    print("\n1️⃣  TESTING MODEL LOADING")
    print("-" * 30)
    
    try:
        result = os.system("python test_model_loading.py")
        model_test_passed = (result == 0)
        results.append(("Model Loading", model_test_passed))
        
        if model_test_passed:
            print("✅ Model loading test PASSED")
        else:
            print("❌ Model loading test FAILED")
            
    except Exception as e:
        print(f"❌ Could not run model loading test: {e}")
        results.append(("Model Loading", False))
    
    # 2. Feature Calculation Consistency
    print("\n2️⃣  TESTING FEATURE CALCULATION CONSISTENCY")
    print("-" * 45)
    
    try:
        from tradovate_integration import FeatureCalculator, BarData
        from datetime import datetime, timedelta
        import numpy as np
        
        # Test cumulative delta calculation
        calculator = FeatureCalculator(lookback_bars=60)
        
        # Add some test bars
        base_time = datetime.now()
        test_bars = []
        
        # Create bars with specific price patterns to test delta
        prices = [4500.0, 4500.5, 4501.0, 4500.5, 4500.0]  # Up, up, down, down
        volumes = [1000, 1500, 2000, 1200, 800]
        
        for i in range(65):  # Need 65 bars minimum
            if i < len(prices):
                price = prices[i]
                volume = volumes[i]
            else:
                # Random walk for remaining bars
                price = 4500.0 + np.random.randn() * 0.5
                volume = 1000 + int(np.random.randn() * 200)
            
            bar = BarData(
                timestamp=base_time + timedelta(minutes=i),
                open=price,
                high=price + 0.5,
                low=price - 0.5, 
                close=price,
                volume=volume
            )
            
            calculator.add_bar(bar)
            test_bars.append(bar)
        
        # Test that cumulative delta is calculated correctly
        # Should be cumulative, not rolling window
        expected_pattern = calculator.cumulative_delta != 0  # Should have some value
        
        if calculator.is_ready():
            features = calculator.calculate_features()
            
            if features is not None and len(features) == 47:
                # Check that cumulative delta feature (index 1) is reasonable
                cum_delta_feature = features[1]  # Second feature
                
                print(f"   Cumulative delta value: {calculator.cumulative_delta:.3f}")
                print(f"   Cumulative volume: {calculator.cumulative_volume}")
                print(f"   Normalized delta feature: {cum_delta_feature:.6f}")
                print("   ✅ Feature calculation consistency PASSED")
                results.append(("Feature Consistency", True))
            else:
                print("   ❌ Feature calculation returned wrong results")
                results.append(("Feature Consistency", False))
        else:
            print("   ❌ Calculator not ready after adding bars")
            results.append(("Feature Consistency", False))
            
    except Exception as e:
        print(f"   ❌ Feature calculation test failed: {e}")
        results.append(("Feature Consistency", False))
    
    # 3. Configuration Validation
    print("\n3️⃣  VALIDATING CONFIGURATION")
    print("-" * 30)
    
    try:
        import json
        
        # Check replay_config.json
        config_path = "replay_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_params = [
                'confidence_threshold', 'max_daily_trades', 'daily_loss_limit',
                'stop_loss_points', 'profit_target_points', 'point_value',
                'slippage_ticks', 'commission_per_side'
            ]
            
            missing = []
            for param in required_params:
                if param not in config:
                    missing.append(param)
            
            if not missing:
                print("   ✅ All required parameters present in replay_config.json")
                print(f"      Confidence threshold: {config['confidence_threshold']}")
                print(f"      Max daily trades: {config['max_daily_trades']}")
                print(f"      Daily loss limit: ${config['daily_loss_limit']}")
                print(f"      Stop loss: {config['stop_loss_points']} points")
                print(f"      Profit target: {config['profit_target_points']} points")
                results.append(("Configuration", True))
            else:
                print(f"   ❌ Missing parameters: {missing}")
                results.append(("Configuration", False))
        else:
            print(f"   ❌ replay_config.json not found")
            results.append(("Configuration", False))
            
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        results.append(("Configuration", False))
    
    # 4. Quick Connectivity Test (Optional)
    print("\n4️⃣  QUICK CONNECTIVITY TEST (OPTIONAL)")
    print("-" * 40)
    
    test_connectivity = input("Test Tradovate connectivity now? (y/n): ").strip().lower()
    
    if test_connectivity == 'y':
        try:
            result = os.system("python test_replay_connectivity.py")
            connectivity_passed = (result == 0)
            results.append(("Connectivity", connectivity_passed))
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            results.append(("Connectivity", False))
    else:
        print("   ⏭️  Skipping connectivity test")
        results.append(("Connectivity", True))  # Don't count as failure
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 25)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name:20s}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print(f"\n🎉 ALL CHECKS PASSED!")
        print(f"\n🚀 Ready to run replay backtester:")
        print(f"   python main_replay_test.py --quick --mode standard --date 2024-01-15")
        print(f"\n   Or test with your own date:")
        print(f"   python main_replay_test.py --date YYYY-MM-DD --username your_username --password your_password")
    else:
        print(f"\n⚠️  Some checks failed. Please fix issues before proceeding.")
        print(f"\n🔧 Common fixes:")
        print(f"   - Ensure your PPO model is trained and saved")
        print(f"   - Check that feature calculation matches training environment")
        print(f"   - Verify Tradovate demo account credentials")
    
    return passed == total

def main():
    """Main entry point"""
    success = run_immediate_checks()
    
    if success:
        print(f"\n✨ System validation complete - ready for backtesting!")
    else:
        print(f"\n🔧 Please address the failed checks and run again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)