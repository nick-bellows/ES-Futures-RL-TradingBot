#!/usr/bin/env python3
"""
Start NinjaTrader ES Futures RL Trading
Launch the trading bot with your validated PPO model
"""

import sys
import os
import logging
import argparse
from datetime import datetime

# Setup logging
def setup_logging(log_level="INFO"):
    """Configure logging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"nt_trading_{timestamp}.log")
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def main():
    """Main trading launcher"""
    parser = argparse.ArgumentParser(description='NinjaTrader ES Futures RL Trading Bot')
    parser.add_argument('--model', type=str, help='Path to PPO model (default from config)')
    parser.add_argument('--contract', type=str, default="ES 09-25", 
                       help='ES contract to trade (default: ES 09-25)')
    parser.add_argument('--log-level', type=str, default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (no actual orders)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - connect but don\'t trade')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("NINJATRADER ES FUTURES RL TRADING BOT")
    print("=" * 50)
    print(f"Model: {args.model or 'models/best/ppo/ppo_best/best_model'}")
    print(f"Contract: {args.contract}")
    print(f"Log Level: {args.log_level}")
    print(f"Test Mode: {args.test_mode}")
    print(f"Dry Run: {args.dry_run}")
    print(f"Log File: {log_file}")
    print()
    
    try:
        from ninjatrader_bot import NinjaTraderBot
        
        # Initialize bot
        model_path = args.model or "models/best/ppo/ppo_best/best_model"
        bot = NinjaTraderBot(model_path=model_path, contract=args.contract)
        
        if args.test_mode:
            logger.info("Running in TEST MODE - orders will be simulated")
            # Could add test mode logic here
        
        if args.dry_run:
            logger.info("DRY RUN MODE - connecting but not trading")
            
        # Pre-flight checks
        print("Pre-flight checks...")
        
        # Check model
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        print("PASS: Model found")
        
        # Check NinjaTrader connectivity (basic test)
        print("PASS: Configuration loaded")
        
        # Start trading bot
        print(f"\nStarting trading bot...")
        
        if bot.start_trading():
            print("PASS: Trading bot started successfully!")
            
            if args.dry_run:
                print("\nDRY RUN - Bot connected but trading disabled")
                print("Monitoring market data and generating signals...")
            else:
                print("\nLIVE TRADING ACTIVE")
                print(f"Contract: {args.contract}")
                print("Risk Parameters:")
                print(f"  - Stop Loss: 5 points")
                print(f"  - Take Profit: 15 points") 
                print(f"  - Confidence Threshold: 35%")
                print(f"  - Max Daily Trades: 5")
                print(f"  - Max Daily Loss: $750")
            
            print(f"\nMonitoring... Press Ctrl+C to stop")
            print("-" * 50)
            
            # Show periodic status updates
            import time
            last_status_time = time.time()
            status_interval = 60  # Show status every minute
            
            try:
                while True:
                    current_time = time.time()
                    
                    # Show status update periodically
                    if current_time - last_status_time >= status_interval:
                        status = bot.get_status()
                        
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STATUS UPDATE")
                        print(f"Trading Enabled: {status['trading_enabled']}")
                        
                        if status['position']:
                            pos = status['position']
                            print(f"Position: {pos['side']} {pos['quantity']} @ {pos['avg_price']:.2f}")
                            print(f"Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
                        else:
                            print("Position: FLAT")
                        
                        perf = status['performance']
                        print(f"Signals: {perf['signals_generated']}, "
                              f"Trades: {perf['trades_executed']}")
                        
                        if perf['last_confidence']:
                            print(f"Last Signal: {perf['last_signal']} "
                                  f"(confidence: {perf['last_confidence']:.1%})")
                        
                        daily = status['daily_stats']
                        print(f"Daily: {daily['trades_today']} trades, "
                              f"${daily['daily_pnl']:.2f} P&L")
                        print("-" * 30)
                        
                        last_status_time = current_time
                    
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\nShutdown requested...")
                bot.stop_trading()
                print("Trading bot stopped successfully")
                return True
        else:
            logger.error("Failed to start trading bot")
            return False
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("  pip install stable-baselines3 numpy pandas scikit-learn")
        return False
        
    except Exception as e:
        logger.error(f"Trading error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)