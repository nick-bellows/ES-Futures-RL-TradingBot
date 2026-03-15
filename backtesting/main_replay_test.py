#!/usr/bin/env python3
"""
Tradovate Replay Backtest Runner
Main script to run comprehensive replay testing of the trading model
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import getpass
from replay_test_suite import (
    run_replay_tests, 
    run_stress_tests, 
    run_parameter_sensitivity_tests,
    create_comprehensive_report,
    print_detailed_analysis
)

def load_config():
    """Load configuration from file or create default"""
    config_file = "replay_config.json"
    
    default_config = {
        "model_path": "models/best/ppo/ppo_best/best_model",
        "output_dir": "replay_results",
        "test_modes": ["standard", "stress", "sensitivity"],
        "confidence_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "replay_settings": {
            "speed": 10,
            "contract_id": 0,
            "timeout_minutes": 30
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            config = default_config
    else:
        config = default_config
        # Save default config for future use
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default configuration: {config_file}")
    
    return config

def validate_model_path(model_path):
    """Validate that the model exists"""
    if os.path.exists(f"{model_path}.zip"):
        return True
    elif os.path.exists(model_path):
        return True
    else:
        print(f"Error: Model not found at {model_path}")
        print("Available models:")
        
        # Look for models in common locations
        model_dirs = ["models", "models/best", "models/best/ppo"]
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                print(f"\nIn {model_dir}:")
                for item in os.listdir(model_dir):
                    if item.endswith('.zip') or os.path.isdir(os.path.join(model_dir, item)):
                        print(f"  - {item}")
        return False

def setup_output_directory(output_dir):
    """Create output directory and subdirectories"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    print(f"Output directory prepared: {output_dir}")

def get_credentials(args):
    """Get Tradovate credentials"""
    if args.username and args.password:
        return args.username, args.password
    
    print("\nTradovate Demo Account Credentials:")
    print("(Make sure you're using DEMO credentials, not live!)")
    
    username = input("Username: ").strip()
    if not username:
        print("Username is required")
        sys.exit(1)
    
    password = getpass.getpass("Password: ")
    if not password:
        print("Password is required")
        sys.exit(1)
    
    return username, password

def save_results(results_df, stress_df, sensitivity_df, output_dir):
    """Save all results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save DataFrames
    if not results_df.empty:
        results_file = os.path.join(output_dir, "data", f"replay_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Main results saved: {results_file}")
    
    if stress_df is not None and not stress_df.empty:
        stress_file = os.path.join(output_dir, "data", f"stress_results_{timestamp}.csv")
        stress_df.to_csv(stress_file, index=False)
        print(f"Stress results saved: {stress_file}")
    
    if sensitivity_df is not None and not sensitivity_df.empty:
        sensitivity_file = os.path.join(output_dir, "data", f"sensitivity_results_{timestamp}.csv")
        sensitivity_df.to_csv(sensitivity_file, index=False)
        print(f"Sensitivity results saved: {sensitivity_file}")
    
    return timestamp

def generate_summary_report(results_df, stress_df, sensitivity_df, config, timestamp, output_dir):
    """Generate a comprehensive summary report"""
    report_file = os.path.join(output_dir, "reports", f"replay_summary_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        # Redirect stdout to file temporarily
        original_stdout = sys.stdout
        sys.stdout = f
        
        print("TRADOVATE REPLAY BACKTEST SUMMARY REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {config['model_path']}")
        print(f"Test ID: {timestamp}")
        print("\n")
        
        # Print detailed analysis
        print_detailed_analysis(results_df, stress_df, sensitivity_df)
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"Summary report saved: {report_file}")
    return report_file

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run Tradovate Replay Backtests')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--username', type=str, help='Tradovate demo username')
    parser.add_argument('--password', type=str, help='Tradovate demo password')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--mode', type=str, choices=['standard', 'stress', 'sensitivity', 'all'], 
                       default='all', help='Test mode to run')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quick', action='store_true', help='Run quick test (fewer dates)')
    
    args = parser.parse_args()
    
    print("TRADOVATE REPLAY BACKTEST SYSTEM")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.model:
        config['model_path'] = args.model
    if args.output:
        config['output_dir'] = args.output
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Validate model
    if not validate_model_path(config['model_path']):
        sys.exit(1)
    
    # Setup output directory
    setup_output_directory(config['output_dir'])
    
    # Get credentials
    username, password = get_credentials(args)
    
    print(f"\nModel: {config['model_path']}")
    print(f"Output: {config['output_dir']}")
    print(f"Test Mode: {args.mode}")
    
    # Initialize result containers
    results_df = pd.DataFrame()
    stress_df = None
    sensitivity_df = None
    
    try:
        # Run standard tests
        if args.mode in ['standard', 'all']:
            print(f"\nRunning Standard Market Condition Tests...")
            results_df = run_replay_tests(config['model_path'], username, password)
            
            if results_df.empty:
                print("ERROR: No results from standard tests")
            else:
                print(f"✅ Standard tests completed: {len(results_df)} test dates")
        
        # Run stress tests
        if args.mode in ['stress', 'all'] and not args.quick:
            print(f"\n⚡ Running Stress Tests...")
            stress_df = run_stress_tests(config['model_path'], username, password)
            
            if stress_df is None or stress_df.empty:
                print("ERROR: No results from stress tests")
            else:
                print(f"✅ Stress tests completed: {len(stress_df)} challenging conditions")
        
        # Run sensitivity tests
        if args.mode in ['sensitivity', 'all'] and not args.quick:
            print(f"\n🎛️  Running Parameter Sensitivity Tests...")
            sensitivity_df = run_parameter_sensitivity_tests(config['model_path'], username, password)
            
            if sensitivity_df is None or sensitivity_df.empty:
                print("ERROR: No results from sensitivity tests")
            else:
                print(f"✅ Sensitivity tests completed: {len(sensitivity_df)} confidence levels")
        
        # Save results
        if not results_df.empty or (stress_df is not None and not stress_df.empty):
            timestamp = save_results(results_df, stress_df, sensitivity_df, config['output_dir'])
            
            # Generate visualizations
            if not args.skip_plots:
                print(f"\n📈 Generating visualizations...")
                try:
                    fig = create_comprehensive_report(results_df, stress_df, sensitivity_df)
                    plot_file = os.path.join(config['output_dir'], "reports", f"replay_charts_{timestamp}.png")
                    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                    print(f"Charts saved: {plot_file}")
                except Exception as e:
                    print(f"Warning: Could not generate plots: {e}")
            
            # Generate summary report
            report_file = generate_summary_report(results_df, stress_df, sensitivity_df, 
                                                config, timestamp, config['output_dir'])
            
            # Print final summary to console
            print(f"\nFINAL SUMMARY")
            print("=" * 30)
            
            if not results_df.empty:
                total_pnl = results_df['total_pnl'].sum()
                avg_win_rate = results_df['win_rate'].mean()
                profitable_days = (results_df['total_pnl'] > 0).sum()
                
                print(f"Total P&L: ${total_pnl:.2f}")
                print(f"Average Win Rate: {avg_win_rate:.1%}")
                print(f"Profitable Days: {profitable_days}/{len(results_df)}")
                
                if total_pnl > 1000:
                    print("STRONG PERFORMANCE - Consider live testing")
                elif total_pnl > 0:
                    print("MODEST PERFORMANCE - Review and optimize")
                else:
                    print("NEEDS IMPROVEMENT - Model requires work")
            
            print(f"\nFull report: {report_file}")
            print(f"All files in: {config['output_dir']}")
            
        else:
            print("ERROR: No test results generated")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()