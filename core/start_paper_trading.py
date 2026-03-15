#!/usr/bin/env python3
"""
Paper Trading Launcher
Starts live paper trading with your validated model
"""

import os
import sys
import logging
from datetime import datetime
from tradovate_integration import PaperTrader, TradovateConfig

def setup_logging():
    """Configure logging for paper trading"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"paper_trading_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def main():
    """Start paper trading"""
    print("ES FUTURES PAPER TRADING")
    print("=" * 40)
    print("Starting live paper trading with validated model")
    print()
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    print(f"Logging to: {log_file}")
    print()
    
    try:
        # Load configuration
        config = TradovateConfig()
        
        # Validate critical components
        print("Pre-flight checks...")
        
        # Check model
        model_path = config.model_path
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            return False
        print(f"✓ Model found: {model_path}")
        
        # Check scaler (optional)
        if os.path.exists(config.scaler_path):
            print(f"✓ Scaler found: {config.scaler_path}")
        else:
            print(f"⚠ No scaler found - using raw features")
        
        # Show trading parameters
        print(f"\nTrading Parameters:")
        print(f"  Stop Loss: {config.trading_params.stop_loss_points} points")
        print(f"  Take Profit: {config.trading_params.take_profit_points} points")
        print(f"  Confidence Threshold: {config.trading_params.confidence_threshold}")
        print(f"  Max Daily Loss: ${config.trading_params.max_daily_loss}")
        print(f"  Max Trades/Day: {config.trading_params.max_trades_per_day}")
        
        # Initialize paper trader
        print(f"\nInitializing paper trader...")
        paper_trader = PaperTrader(config)
        
        # Get credentials
        username = input("Tradovate Username: ").strip()
        if not username:
            print("Username required")
            return False
        
        import getpass
        password = getpass.getpass("Tradovate Password: ")
        if not password:
            print("Password required") 
            return False
        
        print(f"\nStarting paper trading...")
        print(f"Press Ctrl+C to stop")
        print("-" * 40)
        
        # Start paper trading
        success = paper_trader.start_trading(username, password)
        
        if success:
            print("Paper trading started successfully!")
            
            # Keep running until user stops
            try:
                paper_trader.run_forever()
            except KeyboardInterrupt:
                print("\nStopping paper trading...")
                paper_trader.stop_trading()
                print("Paper trading stopped.")
        else:
            print("Failed to start paper trading")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Paper trading error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)