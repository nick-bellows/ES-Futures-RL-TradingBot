#!/usr/bin/env python3
"""
ES Futures RL Trading Bot - Main Entry Point
Uses the new project-based bridge system for real-time trading
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import new project-based bridge components
from ninjatrader_bridge.market_data_bridge import ProjectMarketDataBridge
from ninjatrader_bridge.signal_writer import ProjectSignalWriter
from ninjatrader_bridge.execution_monitor import ProjectExecutionMonitor

# Import contract management
from config.contract_config import display_contract_info, get_contract_status, determine_current_contract
from utils.contract_roller import ContractRoller

# Import ML model components
from stable_baselines3 import PPO
from src.features.qc_features import QCFeatureEngine
from collections import deque
import numpy as np
import pandas as pd
import joblib
import torch

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def display_contract_status():
    """Display current contract status and roll information"""
    print("\n" + "="*50)
    print("ES FUTURES CONTRACT STATUS")
    print("="*50)
    
    try:
        # Get current contract info
        current_contract = determine_current_contract()
        contract_status = get_contract_status(current_contract)
        
        print(display_contract_info(current_contract))
        
        # Check for roll alerts
        roller = ContractRoller()
        alert = roller.generate_roll_alert()
        
        if alert:
            print(f"\n🚨 ROLL ALERT ({alert['urgency']}): {alert['message']}")
            if alert['warnings']:
                for warning in alert['warnings']:
                    print(f"   ⚠️  {warning}")
        
        print("="*50)
        
        return contract_status
        
    except Exception as e:
        print(f"ERROR checking contract status: {e}")
        return None

def load_ppo_model():
    """Load the trained PPO model and feature scaler"""
    try:
        model_path = Path("models/best/ppo/ppo_best/best_model.zip")
        scaler_path = Path("models/feature_scaler.pkl")
        
        if not model_path.exists():
            print(f"ERROR: PPO model not found at {model_path}")
            return None, None
        
        print(f"Loading PPO model from {model_path}")
        model = PPO.load(str(model_path))
        
        # Count parameters
        param_count = sum(p.numel() for p in model.policy.parameters())
        print(f"[OK] PPO model loaded successfully ({param_count:,} parameters)")
        
        # Load feature scaler if available
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"[OK] Feature scaler loaded from {scaler_path}")
        else:
            print(f"[WARN] Feature scaler not found at {scaler_path}")
        
        return model, scaler
        
    except Exception as e:
        print(f"ERROR loading PPO model: {e}")
        return None, None

def build_observation(feature_history, current_position=0, daily_trades=0):
    """Build observation vector for PPO model (2,830 dimensions)"""
    if len(feature_history) < 60:
        return None
    
    # Get last 60 bars of features (convert deque to list for slicing)
    recent_features = list(feature_history)[-60:]
    
    # Flatten 60 bars × 47 features = 2,820
    flat_features = np.array(recent_features).flatten()
    
    # Add 10 position features
    position_features = np.array([
        float(current_position),        # -1, 0, 1 for short, flat, long
        float(daily_trades) / 5.0,      # normalized daily trades
        0.0,                           # daily P&L (placeholder)
        1.0,                           # balance ratio
        0.0,                           # unrealized P&L
        0.0,                           # stop loss price ratio
        0.0,                           # target price ratio  
        0.0,                           # trailing stop ratio
        0.0,                           # episode trades normalized
        0.0                            # episode P&L normalized
    ])
    
    # Combine to 2,830 dimensions
    observation = np.concatenate([flat_features, position_features])
    
    if observation.shape[0] != 2830:
        print(f"WARNING: Observation shape is {observation.shape[0]}, expected 2,830")
        return None
    
    return observation.astype(np.float32)

def check_prerequisites():
    """Check if all prerequisites are met before starting bot"""
    print("Checking prerequisites...")
    
    # Check if bridge directory exists
    bridge_dir = Path("data/bridge")
    if not bridge_dir.exists():
        print("ERROR: Bridge directory not found")
        print("   Run: python utils/setup_bridge_clean.py")
        return False
    
    # Check if market data file exists and is recent
    market_data_file = bridge_dir / "market_data.csv"
    if not market_data_file.exists():
        print("ERROR: Market data file not found")
        print("   Make sure NinjaScript ESDataBridge is running")
        return False
    
    # Check data freshness
    try:
        stat = market_data_file.stat()
        age_seconds = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds()
        
        if age_seconds > 60:
            print(f"WARNING: Market data is {age_seconds:.0f} seconds old")
            print("   Check if NinjaTrader ESDataBridge is active")
            return False
        else:
            print(f"OK: Market data is fresh ({age_seconds:.1f}s old)")
    except Exception as e:
        print(f"ERROR: Error checking market data: {e}")
        return False
    
    print("OK: All prerequisites met")
    return True

def run_trading_bot(duration_minutes=None, dry_run=False):
    """
    Run the main trading bot
    
    Args:
        duration_minutes: How long to run (None = indefinite)
        dry_run: If True, don't send real trading signals
    """
    
    logger = logging.getLogger(__name__)
    
    print("Starting ES Futures RL Trading Bot")
    print("=" * 50)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"Duration: {duration_minutes or 'Indefinite'} minutes")
    print()
    
    # Initialize bridge components
    try:
        print("Initializing bridge components...")
        market_bridge = ProjectMarketDataBridge()
        signal_writer = ProjectSignalWriter()
        execution_monitor = ProjectExecutionMonitor()
        
        # Load PPO model
        print("\nLoading PPO model...")
        ppo_model, feature_scaler = load_ppo_model()
        if ppo_model is None:
            print("ERROR: Failed to load PPO model. Exiting.")
            return False
        
        # Initialize feature engine and history
        print("Initializing feature engine...")
        feature_engine = QCFeatureEngine()
        price_history = deque(maxlen=200)  # Keep extra for feature calculation
        feature_history = deque(maxlen=60)  # 60 bars for model input
        
        # Trading state
        current_position = 0  # -1=short, 0=flat, 1=long
        daily_trades = 0
        signal_count = 0
        prediction_count = 0
        
        print("[OK] All components initialized successfully")
        
        # Start market data monitoring
        if not market_bridge.start_monitoring():
            print("ERROR: Failed to start market data monitoring")
            return False
            
        # Give monitoring a moment to read initial data
        print("Waiting for market data...")
        time.sleep(2.0)
        
        # Check what data we have available
        initial_tick = market_bridge.get_latest_tick()
        if initial_tick:
            age = market_bridge.get_data_age_seconds()
            print(f"Market data found: {initial_tick.last:.2f} (age: {age:.1f}s)")
            
            if age < 30.0:  # Accept data up to 30 seconds old for startup
                print(f"OK: Market data ready for trading")
            elif age < 120.0:  # Up to 2 minutes for testing
                print(f"WARNING: Using older data for testing ({age:.1f}s old)")
            else:
                print(f"WARNING: Data is very stale ({age:.1f}s old)")
                if not dry_run:
                    print("ERROR: Data too old for live trading - use --dry-run for testing")
                    return False
                else:
                    print("Proceeding in dry-run mode with stale data")
        else:
            print("ERROR: No market data available - check NinjaScript ESDataBridge")
            return False
            
        print("OK: Bridge components initialized and monitoring started")
        
    except Exception as e:
        print(f"ERROR: Error initializing bridge: {e}")
        return False
    
    # Main trading loop
    try:
        start_time = time.time()
        tick_count = 0
        signal_count = 0
        
        print("\nStarting trading loop...")
        print("Real-time Status:")
        print("-" * 50)
        
        while True:
            current_time = time.time()
            
            # Check duration limit
            if duration_minutes and (current_time - start_time) > (duration_minutes * 60):
                print(f"\nDuration limit reached ({duration_minutes} minutes)")
                break
            
            try:
                # Get latest market data
                current_tick = market_bridge.get_latest_tick()
                
                if current_tick:
                    data_age = market_bridge.get_data_age_seconds()
                    
                    # Handle infinite age (parsing error)
                    if data_age == float('inf') or data_age < 0:
                        print(f"WARNING: Invalid data age ({data_age}), skipping this tick")
                        time.sleep(0.5)
                        continue
                    
                    # Accept data based on mode (fresh for live, older for testing)
                    max_age = 30.0 if dry_run else 10.0
                    if data_age < max_age:
                        tick_count += 1
                        current_price = current_tick.last
                        
                        # Add price data to history (OHLCV format for features)
                        price_data = {
                            'Open': current_tick.open or current_price,
                            'High': current_tick.high or current_price,
                            'Low': current_tick.low or current_price,
                            'Close': current_price,
                            'Volume': current_tick.volume or 0,
                            'timestamp': datetime.now()
                        }
                        price_history.append(price_data)
                        
                        # Calculate features when we have enough history
                        if len(price_history) >= 100:  # Need minimum data for indicators
                            try:
                                # Convert to DataFrame for feature calculation
                                df = pd.DataFrame(list(price_history))
                                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()
                                
                                # Calculate features for latest bar
                                features_df = feature_engine.calculate_all_features(df)
                                if not features_df.empty:
                                    # Extract numeric features only (excluding OHLCV and timestamps)
                                    feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                                    numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
                                    
                                    if len(numeric_features.columns) >= 40:  # Reasonable number of features
                                        latest_features = numeric_features.iloc[-1].values
                                        feature_history.append(latest_features)
                                        
                            except Exception as e:
                                print(f"WARNING: Feature calculation error: {e}")
                        
                        # Print status every 60 seconds (reduced noise)
                        history_status = f"History: {len(feature_history)}/60"
                        if tick_count % 60 == 0:
                            print(f"{datetime.now().strftime('%H:%M:%S')} | "
                                  f"Price: {current_price:.2f} | "
                                  f"Predictions: {prediction_count} | "
                                  f"Signals: {signal_count}")
                        
                        # Generate predictions when we have full history
                        if len(feature_history) >= 60:
                            try:
                                # Build observation for PPO model
                                observation = build_observation(feature_history, current_position, daily_trades)
                                
                                if observation is not None:
                                    # Get model prediction with action probabilities
                                    action_idx, _ = ppo_model.predict(observation, deterministic=False)
                                    prediction_count += 1
                                    
                                    # Get action probabilities for confidence calculation
                                    try:
                                        obs_tensor = observation.reshape(1, -1)
                                        action_probs = ppo_model.policy.get_distribution(obs_tensor).distribution.probs
                                        confidence = float(action_probs[0, int(action_idx)])
                                    except:
                                        # Fallback confidence based on action frequency
                                        confidence = 0.45 if int(action_idx) != 0 else 0.35
                                    
                                    # Map actions: {0: "HOLD", 1: "BUY", 2: "SELL"}
                                    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                                    action = action_map.get(int(action_idx), "HOLD")
                                    
                                    # Apply confidence threshold and generate signals
                                    if action != "HOLD" and confidence > 0.35 and not dry_run:
                                        try:
                                            success = signal_writer.write_signal(action, current_price, confidence)
                                            if success:
                                                signal_count += 1
                                                daily_trades += 1
                                                # Update position tracking
                                                if action == "BUY":
                                                    current_position = 1
                                                elif action == "SELL":
                                                    current_position = -1
                                                print(f"[SIGNAL] {action} @ {current_price:.2f} (conf: {confidence:.1%})")
                                            else:
                                                # Get detailed error info from signal writer
                                                stats = signal_writer.get_signal_stats()
                                                print(f"[ERROR] Failed to send {action} signal @ {current_price:.2f}")
                                                print(f"        Signal file: {stats['signal_file_path']}")
                                                print(f"        File exists: {stats['signal_file_exists']}")
                                                print(f"        File writable: {stats['signal_file_writable']}")
                                                print(f"        Write failures: {stats['write_failures']}")
                                                if stats['last_error']:
                                                    print(f"        Last error: {stats['last_error']}")
                                                if stats['last_error_time']:
                                                    print(f"        Error time: {stats['last_error_time']}")
                                        except Exception as e:
                                            print(f"[ERROR] Signal writer exception: {e}")
                                            print(f"        Action: {action}, Price: {current_price:.2f}, Conf: {confidence:.1%}")
                                    
                                    # Log prediction every 500 ticks (much less noise)
                                    if prediction_count % 500 == 0:
                                        print(f"[PREDICTION] #{prediction_count}: {action} (conf: {confidence:.1%})")
                                        
                            except Exception as e:
                                print(f"WARNING: Prediction error: {e}")
                        else:
                            # Still building history (only show every 60 ticks)
                            if tick_count % 60 == 0:
                                print(f"[HISTORY] Building feature history: {len(feature_history)}/60 bars ready")
                        
                        # Check for execution updates
                        if tick_count % 50 == 0:
                            summary = execution_monitor.get_order_summary()
                            if summary['buy_orders'] + summary['sell_orders'] > 0:
                                print(f"Orders: {summary['buy_orders']} buys, {summary['sell_orders']} sells, "
                                      f"{summary['orders_filled']} filled")
                    
                    else:
                        # Data is stale but not critical yet
                        if data_age > 30:  # Warning at 30 seconds
                            print(f"WARNING: Stale data warning: {data_age:.1f}s old")
                            
                            # Emergency stop if data too old (2 minutes)
                            if data_age > 120:
                                print(f"EMERGENCY STOP: Data too old ({data_age:.1f}s)")
                                break
                
                else:
                    # No tick data available at all
                    print(f"WARNING: No market data available")
                    time.sleep(1.0)  # Wait longer when no data
                
                # Sleep briefly to avoid excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print(f"\nBot stopped by user")
                break
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                print(f"WARNING: Trading loop error: {e}")
                time.sleep(1.0)  # Wait before retrying
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            market_bridge.stop_monitoring()
            print("OK: Market data monitoring stopped")
        except:
            pass
    
    # Final summary
    runtime_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print("TRADING SESSION SUMMARY")
    print("=" * 50)
    print(f"Runtime: {runtime_minutes:.1f} minutes")
    print(f"Ticks processed: {tick_count}")
    print(f"Signals sent: {signal_count}")
    
    # Get final execution summary
    try:
        final_summary = execution_monitor.get_order_summary()
        print(f"Total orders: {final_summary['buy_orders'] + final_summary['sell_orders']}")
        print(f"Orders filled: {final_summary['orders_filled']}")
        print(f"Errors: {final_summary['errors']}")
    except:
        pass
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ES Futures RL Trading Bot')
    parser.add_argument('--duration', type=int, help='Run duration in minutes')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no real signals)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--check-only', action='store_true', help='Only check prerequisites and exit')
    parser.add_argument('--contract', type=str, help='Override contract symbol (e.g., "ES 12-24")')
    
    args = parser.parse_args()
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level)
    
    # Display contract status
    contract_status = display_contract_status()
    
    # Handle contract override
    if args.contract:
        print(f"\n⚠️  Contract override specified: {args.contract}")
        print("WARNING: Manual contract override - ensure NinjaScript files match!")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites not met. Please fix issues and try again.")
        return 1
    
    if args.check_only:
        print("\nOK: Prerequisites check passed. Bot ready to run.")
        return 0
    
    # Run the bot
    try:
        success = run_trading_bot(
            duration_minutes=args.duration,
            dry_run=args.dry_run
        )
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())