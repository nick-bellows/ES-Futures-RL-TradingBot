"""
NinjaTrader ES Futures RL Trading Bot
Main bot integration connecting your trained PPO model with NinjaTrader 8
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import deque
import threading
import time

# Import existing components
sys.path.append('.')
from tradovate_integration import FeatureCalculator, BarData as TradovateBarData
from stable_baselines3 import PPO

# Import NinjaTrader bridge
from ninjatrader_bridge import NTConfig, NTConnector, NTDataFeed, NTOrderManager
from ninjatrader_bridge.nt_data_feed import BarData
from ninjatrader_bridge.nt_order_manager import OrderAction, PositionSide

class NinjaTraderBot:
    """
    ES Futures RL Trading Bot with NinjaTrader 8 Integration
    
    Uses your validated PPO model with exact feature calculation
    from training to trade ES futures through NinjaTrader.
    """
    
    def __init__(self, model_path: str, contract: str = "ES 09-25"):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = NTConfig()
        self.config.trading.instrument = contract
        
        # NinjaTrader components
        self.connector = NTConnector(self.config)
        self.data_feed = NTDataFeed(self.config, self.connector)
        self.order_manager = NTOrderManager(self.config, self.connector)
        
        # Trading model
        self.model = None
        self.scaler = None
        self.model_path = model_path
        
        # Feature calculation (using existing validated calculator)
        self.feature_calculator = FeatureCalculator(lookback_bars=self.config.trading.lookback_bars)
        
        # Historical feature storage for model input (60 bars × 47 features = 2,820)
        self.feature_history = deque(maxlen=self.config.trading.lookback_bars)  # Store 47-dim features for each bar
        
        # Position tracking for position features (10 additional features)
        self.position_features = np.zeros(10)
        
        # Trading state
        self.trading_enabled = False
        self.current_signal = None
        self.current_confidence = 0.0
        self.last_trade_time = None
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_attempted = 0
        self.trades_executed = 0
        
        # Safety monitoring
        self.monitoring_thread = None
        self.running = False
        
        # Initialize components
        self._initialize_model()
        self._setup_callbacks()
        
        # Clear any old signals on startup
        self._clear_signal_file()
    
    def _initialize_model(self) -> bool:
        """Load and initialize the trading model"""
        try:
            self.logger.info(f"Loading PPO model from {self.model_path}")
            
            # Load model
            if os.path.exists(self.model_path + ".zip"):
                self.model = PPO.load(self.model_path)
            elif os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            self.logger.info(f"Model loaded successfully: {total_params:,} parameters")
            
            # Load scaler if available
            scaler_path = self.config.scaler_path
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Feature scaler loaded: {len(self.scaler.mean_)} features")
            else:
                self.logger.warning("No feature scaler found - using raw features")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            return False
    
    def _setup_callbacks(self) -> None:
        """Setup data feed and order callbacks"""
        # Bar update callback - this is where trading decisions are made
        self.data_feed.add_bar_callback(self.on_bar_update)
        
        # Order status callbacks
        self.order_manager.add_order_callback(self.on_order_update)
        self.order_manager.add_position_callback(self.on_position_update)
    
    def _clear_signal_file(self):
        """Clear any old signals on startup"""
        import os
        signal_file = r"C:\temp\bot_signals.txt"
        try:
            if os.path.exists(signal_file):
                os.remove(signal_file)
                self.logger.info("Cleared old signal file")
        except Exception as e:
            self.logger.debug(f"Could not clear signal file: {e}")
    
    def write_signal_for_ninjatrader(self, action, price, confidence):
        """Write signal file for NinjaScript to read"""
        import os
        from datetime import datetime
        
        signal_dir = r"C:\temp"
        signal_file = os.path.join(signal_dir, "bot_signals.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(signal_dir, exist_ok=True)
        
        # Convert numeric action to text
        # 0 = HOLD, 1 = BUY, 2 = SELL
        if action == 1:
            action_text = "BUY"
        elif action == 2:
            action_text = "SELL"
        else:
            return  # Don't write HOLD signals
        
        # Write signal file
        signal = f"{action_text},{price},{confidence:.4f},{datetime.now()}"
        with open(signal_file, 'w') as f:
            f.write(signal)
        
        self.logger.info(f"Signal written to NinjaScript: {signal}")
        print(f"[NINJASCRIPT SIGNAL] {action_text} at {price} (confidence: {confidence:.2%})")
    
    def start_trading(self) -> bool:
        """Start the trading bot"""
        try:
            self.logger.info("Starting NinjaTrader ES Futures RL Trading Bot")
            
            # Connect to NinjaTrader
            if not self.connector.connect():
                self.logger.error("Failed to connect to NinjaTrader")
                return False
            
            # Test API endpoints
            endpoint_results = self.connector.test_endpoints()
            failed_endpoints = [ep for ep, success in endpoint_results.items() if not success]
            
            if failed_endpoints:
                self.logger.warning(f"Some endpoints failed: {failed_endpoints}")
                # Continue anyway - might work for basic functionality
            
            # Start data streaming
            if not self.data_feed.start_streaming(self.config.trading.instrument):
                self.logger.error("Failed to start data streaming")
                return False
            
            # Start order monitoring
            if not self.order_manager.start_monitoring():
                self.logger.error("Failed to start order monitoring")
                return False
            
            # Wait for initial data
            self.logger.info("Waiting for initial market data...")
            timeout = 200  # 200 seconds to allow 60 bars at 3 seconds each
            start_time = time.time()
            
            while not self.data_feed.is_ready() and (time.time() - start_time) < timeout:
                # Show progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:
                    bars_count = len(self.data_feed.get_bars())
                    self.logger.info(f"Market data progress: {bars_count}/60 bars ({elapsed:.0f}s elapsed)")
                time.sleep(1)
            
            if not self.data_feed.is_ready():
                bars_count = len(self.data_feed.get_bars())
                self.logger.error(f"Failed to receive initial market data - only got {bars_count}/60 bars")
                return False
            
            # CRITICAL SAFETY: Validate first received prices are reasonable
            if not self._validate_initial_market_data():
                self.logger.error("CRITICAL: Initial market data validation failed - TRADING DISABLED")
                return False
            
            # Enable trading
            self.trading_enabled = True
            self.running = True
            
            # Start safety monitoring
            self._start_monitoring()
            
            self.logger.info("Trading bot started successfully!")
            self.logger.info(f"Trading: {self.config.trading.instrument}")
            self.logger.info(f"Stop Loss: {self.config.trading.stop_loss_points} points")
            self.logger.info(f"Take Profit: {self.config.trading.take_profit_points} points")
            self.logger.info(f"Confidence Threshold: {self.config.trading.confidence_threshold}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading: {e}")
            return False
    
    def stop_trading(self) -> None:
        """Stop the trading bot"""
        try:
            self.logger.info("Stopping trading bot...")
            
            # Disable trading
            self.trading_enabled = False
            self.running = False
            
            # Close any open positions
            self.order_manager.close_position()
            
            # Stop monitoring
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            # Stop order monitoring
            self.order_manager.stop_monitoring()
            
            # Stop data streaming
            self.data_feed.stop_streaming()
            
            # Disconnect
            self.connector.disconnect()
            
            self.logger.info("Trading bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    def on_bar_update(self, bar: BarData) -> None:
        """
        Handle new bar updates - main trading logic
        
        This is called every minute with new market data
        """
        try:
            # Convert NinjaTrader bar to Tradovate format for feature calculation
            tradovate_bar = TradovateBarData(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            )
            
            # Add to feature calculator
            self.feature_calculator.add_bar(tradovate_bar)
            
            # Only trade if we have enough data
            if not self.feature_calculator.is_ready() or not self.trading_enabled:
                # If not ready but have some data, still calculate and store features for history
                if self.feature_calculator.get_bar_count() > 0:
                    current_features = self.feature_calculator.calculate_features()
                    if current_features is not None and len(current_features) == 47:
                        self.feature_history.append(current_features.copy())
                return
            
            # Calculate features for current bar (this uses your EXACT training calculation)
            current_features = self.feature_calculator.calculate_features()
            
            if current_features is None or len(current_features) != 47:
                self.logger.warning(f"Invalid features: {current_features is not None and len(current_features) if current_features else None}")
                return
            
            # Store current features in history
            self.feature_history.append(current_features.copy())
            
            # Check if we have enough feature history for model
            if len(self.feature_history) < self.config.trading.lookback_bars:
                self.logger.debug(f"Building feature history: {len(self.feature_history)}/{self.config.trading.lookback_bars}")
                return
            
            # Build model observation: 60 bars × 47 features + 10 position features = 2,830
            observation = self._build_model_observation()
            
            if observation is None or len(observation) != 2830:
                self.logger.warning(f"Invalid observation shape: {len(observation) if observation is not None else None}")
                return
            
            # Apply scaling only to the feature portion (2,820 dims), keep position features raw (10 dims)
            if self.scaler and hasattr(self.scaler, 'transform'):
                try:
                    # Split observation: 2,820 feature dims + 10 position dims
                    feature_portion = observation[:2820].reshape(60, 47)  # Back to (60, 47) matrix
                    position_portion = observation[2820:]  # Last 10 position features
                    
                    # Scale each of the 60 feature vectors individually
                    scaled_features = []
                    for feature_vec in feature_portion:
                        scaled_vec = self.scaler.transform(feature_vec.reshape(1, -1))[0]
                        scaled_features.append(scaled_vec)
                    
                    # Reconstruct observation: scaled features + raw position features
                    scaled_feature_matrix = np.array(scaled_features)  # (60, 47)
                    observation_scaled = np.concatenate([
                        scaled_feature_matrix.flatten(),  # 2,820 scaled features
                        position_portion  # 10 raw position features
                    ])
                    
                except Exception as e:
                    self.logger.warning(f"Scaling failed, using raw observation: {e}")
                    observation_scaled = observation
            else:
                observation_scaled = observation
            
            # Get model prediction (MUST use deterministic=False for stochastic predictions)
            action, _states = self.model.predict(observation_scaled, deterministic=False)
            
            # Extract confidence from policy
            confidence = self._get_prediction_confidence(observation_scaled)
            
            self.current_signal = action
            self.current_confidence = confidence
            self.signals_generated += 1
            
            self.logger.debug(f"Signal: {action}, Confidence: {confidence:.3f}")
            
            # Execute trade if confident enough
            if confidence >= self.config.trading.confidence_threshold:
                self._execute_trade_signal(action, bar.close, confidence)
            
        except Exception as e:
            self.logger.error(f"Error in bar update: {e}")
    
    def _build_model_observation(self) -> Optional[np.ndarray]:
        """
        Build complete model observation: 60 bars × 47 features + 10 position features = 2,830
        """
        try:
            # Ensure we have enough feature history
            if len(self.feature_history) < self.config.trading.lookback_bars:
                return None
            
            # Get the last 60 feature vectors (47 dimensions each)
            feature_matrix = np.array(list(self.feature_history))  # Shape: (60, 47)
            
            # Flatten to get 60 × 47 = 2,820 dimensions
            flattened_features = feature_matrix.flatten()  # Shape: (2820,)
            
            # Update position features (10 dimensions)
            self._update_position_features()
            
            # Combine: 2,820 historical features + 10 position features = 2,830
            observation = np.concatenate([flattened_features, self.position_features])
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Error building model observation: {e}")
            return None
    
    def _update_position_features(self) -> None:
        """
        Update the 10 position features for model input
        """
        try:
            # Get current position and daily stats
            position = self.order_manager.get_position()
            daily_stats = self.order_manager.get_daily_stats()
            
            # Position features (10 dimensions)
            self.position_features[0] = 1.0 if position and position.side.value == "LONG" else 0.0
            self.position_features[1] = 1.0 if position and position.side.value == "SHORT" else 0.0
            self.position_features[2] = position.quantity if position else 0.0
            self.position_features[3] = position.unrealized_pnl / 1000.0 if position else 0.0  # Normalized
            
            # Daily trading stats
            self.position_features[4] = daily_stats['trades_today'] / self.config.trading.max_daily_trades  # Normalized
            self.position_features[5] = daily_stats['daily_pnl'] / self.config.trading.max_daily_loss  # Normalized
            self.position_features[6] = 1.0 if daily_stats['can_trade'] else 0.0
            
            # Time-based features
            current_time = datetime.now()
            hour_of_day = current_time.hour / 24.0  # Normalized hour
            self.position_features[7] = hour_of_day
            
            # Market session (regular hours: 9:30-16:00 ET)
            is_regular_hours = 9.5 <= current_time.hour + current_time.minute/60.0 <= 16.0
            self.position_features[8] = 1.0 if is_regular_hours else 0.0
            
            # Recent signal strength (confidence from last prediction)
            self.position_features[9] = self.current_confidence if self.current_confidence else 0.0
            
        except Exception as e:
            self.logger.error(f"Error updating position features: {e}")
            # Initialize with zeros if error
            self.position_features = np.zeros(10)
    
    def _get_prediction_confidence(self, observation: np.ndarray) -> float:
        """Extract confidence from model prediction"""
        try:
            import torch
            
            with torch.no_grad():
                # Convert observation to tensor (2,830 dimensions)
                obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]
                
                # Get action distribution
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                # Calculate confidence as max probability
                probs = torch.softmax(distribution.distribution.logits, dim=-1)
                confidence = probs.max().item()
                
                return confidence
                
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _validate_trade_price(self, price: float) -> bool:
        """Validate price before executing trades - CRITICAL SAFETY CHECK"""
        try:
            # ES futures should be roughly 6000-7000 range (as of 2025)
            # This prevents trading on completely wrong prices like the 4500 bug
            if price < 5000 or price > 8000:
                self.logger.error(f"CRITICAL: Trade price {price} outside reasonable ES range (5000-8000)")
                return False
            
            # If we have recent market data, check for huge jumps
            current_bar = self.data_feed.get_latest_bar()
            if current_bar and abs(price - current_bar.close) > 100:
                self.logger.error(f"CRITICAL: Price jump too large: {current_bar.close:.2f} -> {price:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade price: {e}")
            return False
    
    def _validate_initial_market_data(self) -> bool:
        """Validate that initial market data contains reasonable prices"""
        try:
            bars = self.data_feed.get_bars(count=5)  # Check last 5 bars
            
            if not bars:
                self.logger.error("No bars available for validation")
                return False
            
            for i, bar in enumerate(bars):
                # Check each price component
                prices = [bar.open, bar.high, bar.low, bar.close]
                
                for price_type, price in zip(['open', 'high', 'low', 'close'], prices):
                    # ES futures should be roughly 6000-7000 range (as of 2025)
                    if price < 5000 or price > 8000:
                        self.logger.error(f"CRITICAL: Bar {i} {price_type} price {price:.2f} outside ES range (5000-8000)")
                        return False
                
                # Basic OHLC validation
                if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
                    self.logger.error(f"CRITICAL: Bar {i} has invalid OHLC: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}")
                    return False
            
            # Log successful validation
            latest_bar = bars[-1]
            self.logger.info(f"✓ Market data validation PASSED - Latest price: {latest_bar.close:.2f}")
            self.logger.info(f"✓ Price range validated: ES futures in expected range (5000-8000)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating initial market data: {e}")
            return False
    
    def _execute_trade_signal(self, action: int, current_price: float, confidence: float) -> None:
        """Execute trading signal"""
        try:
            # Check if we can trade
            daily_stats = self.order_manager.get_daily_stats()
            if not daily_stats['can_trade']:
                self.logger.info("Trading disabled - daily limits reached")
                return
            
            # CRITICAL SAFETY: Prevent rapid-fire trading (minimum 60 seconds between trades)
            if self.last_trade_time:
                time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds()
                min_trade_interval = 60  # 60 seconds minimum
                
                if time_since_last_trade < min_trade_interval:
                    self.logger.info(f"Trade blocked - too soon after last trade ({time_since_last_trade:.1f}s < {min_trade_interval}s)")
                    return
            
            # CRITICAL SAFETY: Validate current price is reasonable
            if not self._validate_trade_price(current_price):
                self.logger.error(f"TRADE BLOCKED - Invalid price: {current_price:.2f}")
                return
            
            # Check current position
            position = self.order_manager.get_position()
            
            # Determine trade action
            # Assuming: action=0 means hold/exit, action=1 means buy, action=2 means sell
            if action == 1:  # Buy signal
                if not position or position.side == PositionSide.FLAT:
                    order_action = OrderAction.BUY
                elif position.side == PositionSide.SHORT:
                    # Close short position first, then go long
                    self.order_manager.close_position()
                    order_action = OrderAction.BUY
                else:
                    # Already long, no action
                    return
            elif action == 2:  # Sell signal
                if not position or position.side == PositionSide.FLAT:
                    order_action = OrderAction.SELL
                elif position.side == PositionSide.LONG:
                    # Close long position first, then go short
                    self.order_manager.close_position()
                    order_action = OrderAction.SELL
                else:
                    # Already short, no action
                    return
            else:
                # Hold/exit signal
                if position and position.side != PositionSide.FLAT:
                    self.order_manager.close_position()
                return
            
            # FIXED: Write signal for NinjaScript instead of using broken AT Interface
            self.write_signal_for_ninjatrader(action, current_price, confidence)
            
            # Update tracking (keep existing logic)
            self.trades_attempted += 1
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"Trade executed: {order_action.value} @ {current_price:.2f} "
                           f"(confidence: {confidence:.1%})")
            self.logger.info("Signal sent to NinjaScript - check NinjaTrader for order execution")
                
        except Exception as e:
            self.logger.error(f"Error executing trade signal: {e}")
    
    def on_order_update(self, order) -> None:
        """Handle order status updates"""
        self.logger.info(f"Order update: {order.order_id[:8]} - {order.status.value}")
        
        if order.status.value in ['FILLED']:
            self.trades_executed += 1
    
    def on_position_update(self, position) -> None:
        """Handle position updates"""
        if position:
            self.logger.info(f"Position: {position.side.value} {position.quantity} @ {position.avg_price:.2f} "
                           f"(P&L: ${position.unrealized_pnl:.2f})")
    
    def _start_monitoring(self) -> None:
        """Start safety monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_worker(self) -> None:
        """Safety monitoring worker"""
        self.logger.info("Safety monitoring started")
        
        while self.running:
            try:
                # Check connection health
                if not self.connector.is_connected():
                    self.logger.error("Lost connection to NinjaTrader")
                    self.trading_enabled = False
                    break
                
                # Check data feed health
                data_stats = self.data_feed.get_statistics()
                if data_stats['consecutive_errors'] > 5:
                    self.logger.error("Too many data feed errors")
                    self.trading_enabled = False
                
                # Check daily limits
                daily_stats = self.order_manager.get_daily_stats()
                if not daily_stats['can_trade'] and self.trading_enabled:
                    self.logger.warning("Daily limits reached - disabling trading")
                    self.trading_enabled = False
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)
        
        self.logger.info("Safety monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        connection_info = self.connector.get_connection_info()
        data_stats = self.data_feed.get_statistics()
        daily_stats = self.order_manager.get_daily_stats()
        position = self.order_manager.get_position()
        
        return {
            'trading_enabled': self.trading_enabled,
            'running': self.running,
            'connection': connection_info,
            'data_feed': data_stats,
            'position': position.to_dict() if position else None,
            'daily_stats': daily_stats,
            'performance': {
                'signals_generated': self.signals_generated,
                'trades_attempted': self.trades_attempted,
                'trades_executed': self.trades_executed,
                'last_signal': self.current_signal,
                'last_confidence': self.current_confidence,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
            }
        }
    
    def run_forever(self) -> None:
        """Run the bot until stopped"""
        try:
            self.logger.info("Bot running... Press Ctrl+C to stop")
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        finally:
            self.stop_trading()