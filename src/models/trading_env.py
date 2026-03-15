"""
ES Futures RL Trading Environment

Implements a Gymnasium environment for training RL agents on ES futures with:
- $750 daily loss limit
- 5 trades per day maximum  
- 5-point trailing stop loss
- 3:1 risk/reward (15 points profit target, 5 points stop)
- 1-minute ES futures data with 47 QC-compatible features
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1   # Go long (only if flat)
    SELL = 2  # Go short (only if flat) OR close long position


class ESFuturesEnv(gym.Env):
    """ES Futures Trading Environment for RL"""
    
    def __init__(self, 
                 data,  # Can be file path (str) or DataFrame
                 initial_balance: float = 50000,
                 daily_loss_limit: float = 1500,  # TRAINING: Relaxed for more exploration
                 max_daily_trades: int = 10,      # TRAINING: Relaxed for more learning opportunities
                 stop_loss_points: float = 5.0,   # LOCKED: Must match production (3:1 ratio)
                 profit_target_points: float = 15.0,  # LOCKED: Must match production (3:1 ratio)
                 point_value: float = 50,  # ES futures point value
                 lookback_window: int = 60,  # 1-hour lookback
                 reward_scaling: float = 1.0,
                 verbose: bool = False,
                 simple_reward: bool = True):  # TRAINING: Use simplified reward for better learning
        
        super().__init__()
        
        # Load and prepare data - handle both DataFrame and file path
        if isinstance(data, str):
            # It's a file path
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            # It's already a DataFrame
            self.df = data.copy()  # Make a copy to avoid modifying original
        else:
            # Try to read it as a path anyway
            self.df = pd.read_csv(str(data))
        
        # Parse timestamps if Time column exists
        if 'Time' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['Time'], format='%Y%m%d %H:%M')
        elif 'timestamp' not in self.df.columns:
            # If no timestamp column, create index-based timestamps
            self.df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(self.df), freq='1min')
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Get feature columns (exclude OHLCV and Time)
        self.feature_cols = [col for col in self.df.columns 
                           if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']]
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.daily_loss_limit = daily_loss_limit
        self.max_daily_trades = max_daily_trades
        self.stop_loss_points = stop_loss_points
        self.profit_target_points = profit_target_points
        self.point_value = point_value
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.verbose = verbose
        self.simple_reward = simple_reward  # Use simplified reward for training
        
        # Action space: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + position info + account info
        n_features = len(self.feature_cols)
        obs_size = n_features * lookback_window + 10  # +10 for position/account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Log previous episode summary before reset
        if hasattr(self, 'episode_step') and self.episode_step > 0:
            win_rate = self._calculate_win_rate()
            print(f"Episode Summary: Steps={self.episode_step}, Total Trades={len(self.episode_trades)}, "
                  f"P&L=${self.episode_pnl:.2f}, Win Rate={win_rate:.2%}, Balance=${self.balance:.2f}")
            
            if len(self.episode_trades) > 0:
                profitable_trades = sum(1 for trade in self.episode_trades if trade['dollar_pnl'] > 0)
                avg_profit = sum(trade['dollar_pnl'] for trade in self.episode_trades) / len(self.episode_trades)
                print(f"         Profitable: {profitable_trades}/{len(self.episode_trades)}, Avg P&L: ${avg_profit:.2f}")
            else:
                print(f"         WARNING: NO TRADES EXECUTED in {self.episode_step} steps!")
                
            # Action distribution debugging (if we track actions)
            if hasattr(self, 'action_counts'):
                total_actions = sum(self.action_counts.values())
                print(f"         Actions: HOLD={self.action_counts.get(0, 0)}, BUY={self.action_counts.get(1, 0)}, SELL={self.action_counts.get(2, 0)}")
                if total_actions > 0:
                    hold_pct = (self.action_counts.get(0, 0) / total_actions) * 100
                    print(f"         Hold percentage: {hold_pct:.1f}%")
        
        # Reset to random starting point (but ensure enough data for full episode)
        min_start = self.lookback_window
        # CRITICAL FIX: Ensure at least 390 steps available for a full trading day
        episode_length = 390
        max_start = len(self.df) - episode_length - 1
        
        if max_start <= min_start:
            # Dataset too small, use sequential start
            self.current_step = min_start
            if self.verbose:
                print(f"WARNING: Dataset too small for random starts. Using sequential start at {min_start}")
        else:
            self.current_step = np.random.randint(min_start, max_start)
            if self.verbose:
                print(f"Episode starting at step {self.current_step} (can run {len(self.df) - self.current_step} steps)")
        
        # Account state
        self.balance = self.initial_balance
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_day = self.df.iloc[self.current_step]['timestamp'].date()
        
        # Position state
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.entry_step = 0
        self.stop_loss_price = 0.0
        self.profit_target_price = 0.0
        self.trailing_stop_price = 0.0
        
        # Episode tracking
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.done = False
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
        self.episode_step = 0  # Track steps within this episode
        
        # Action diversity tracking
        self.last_action = None
        self.actions_used_this_episode = set()
        self.consecutive_same_actions = 0
        
        # Advanced reward tracking
        self.episode_returns = []  # For Sharpe calculation
        self.last_price = 0.0  # For opportunity cost calculation
        self.flat_time = 0  # Minutes spent flat
        self.position_hold_time = 0  # Minutes in current position
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _calculate_advanced_reward(self, current_price: float, action: int, trade_executed: bool, position_closed: bool, close_reason: str = "") -> float:
        """
        Calculate advanced reward with multiple components - WITH SAFETY CHECKS
        """
        # Initialize components
        pnl_component = 0.0
        sharpe_component = 0.0
        risk_penalty = 0.0
        trade_quality_bonus = 0.0
        opportunity_cost = 0.0
        
        # Safety check for current_price
        if not np.isfinite(current_price) or current_price <= 0:
            if self.verbose:
                print(f"WARNING: Invalid current_price: {current_price}")
            return 0.0
        
        # 1. Scaled PnL Reward (normalize to [-1, 1])
        current_pnl = self._calculate_unrealized_pnl(current_price)
        
        # Safety check for PnL calculation
        if not np.isfinite(current_pnl):
            if self.verbose:
                print(f"WARNING: Invalid current_pnl: {current_pnl}")
            current_pnl = 0.0
            
        if position_closed:
            # Use the actual realized PnL from the last trade
            if len(self.episode_trades) > 0:
                last_trade = self.episode_trades[-1]
                points_pnl = last_trade.get('points_pnl', 0.0)
                if np.isfinite(points_pnl) and points_pnl != 0:
                    pnl_component = points_pnl / 15.0  # Normalize by profit target (15 points)
                else:
                    pnl_component = 0.0
        else:
            # Use unrealized PnL for open positions
            if np.isfinite(current_pnl):
                pnl_component = current_pnl / 15.0  # Normalize by profit target
            else:
                pnl_component = 0.0
            
        # Cap PnL component to [-1, 1] and ensure it's finite
        pnl_component = np.clip(pnl_component, -1.0, 1.0)
        if not np.isfinite(pnl_component):
            if self.verbose:
                print(f"WARNING: Invalid pnl_component after clipping: {pnl_component}")
            pnl_component = 0.0
        
        # 2. Sharpe Ratio Component (30% weight)
        if len(self.episode_returns) >= 20:  # Need some history
            try:
                returns_array = np.array(self.episode_returns[-30:])  # Last 30 returns
                # Safety checks for Sharpe calculation
                if len(returns_array) > 0 and all(np.isfinite(returns_array)):
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    
                    # Prevent division by zero with epsilon
                    if std_return > 1e-8:
                        sharpe = mean_return / (std_return + 1e-8)
                        if np.isfinite(sharpe):
                            sharpe_component = 0.3 * np.clip(sharpe / 2.0, -0.5, 0.5)
                        else:
                            if self.verbose:
                                print(f"WARNING: Invalid Sharpe ratio: {sharpe}")
                            sharpe_component = 0.0
                    else:
                        sharpe_component = 0.0  # No variance, no Sharpe bonus/penalty
                else:
                    if self.verbose:
                        print(f"WARNING: Invalid returns array: {returns_array}")
                    sharpe_component = 0.0
            except Exception as e:
                if self.verbose:
                    print(f"WARNING: Sharpe calculation error: {e}")
                sharpe_component = 0.0
        
        # 3. Position Management Penalties
        # Only penalize if position is still open (not closed)
        if self.position != 0 and not position_closed:
            # Penalize large unrealized losses
            unrealized_loss_dollars = abs(min(0, current_pnl * self.point_value))
            if unrealized_loss_dollars > 250:
                risk_penalty -= 0.1
                
            # Time decay penalty for positions held too long
            self.position_hold_time += 1
            if self.position_hold_time > 60:  # More than 60 minutes
                risk_penalty -= 0.001 * (self.position_hold_time - 60)
        else:
            if self.position == 0:  # Only increment flat time if truly flat
                self.flat_time += 1
            
        # Heavy penalty for daily loss limit violation
        # Only apply if we're actually in a loss situation or haven't just made a profitable trade
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            # Don't penalize if we just closed a profitable position that puts us back in good standing
            if not (position_closed and close_reason == "profit_target" and self.daily_pnl > -self.daily_loss_limit):
                risk_penalty -= 2.0
            
        # Heavy penalty for exceeding daily trade limit
        if self.daily_trades > self.max_daily_trades:  # Changed from >= to >
            risk_penalty -= 1.0
        
        # 4. Trade Quality Bonuses/Penalties
        if position_closed:
            if close_reason == "profit_target":
                trade_quality_bonus += 0.5  # Hit 15-point target
            elif close_reason in ["stop_loss", "trailing_stop"]:
                trade_quality_bonus -= 0.3  # Hit 5-point stop
                
        # Win rate bonus
        if len(self.episode_trades) >= 5:
            winning_trades = sum(1 for t in self.episode_trades if t['points_pnl'] > 0)
            win_rate = winning_trades / len(self.episode_trades)
            if win_rate > 0.4:
                trade_quality_bonus += 0.5 * (win_rate - 0.4)
        
        # 5. Opportunity Cost Penalties
        # Only apply opportunity cost if not in a position-closing event
        if not position_closed:
            if self.last_price > 0:
                price_move = abs(current_price - self.last_price)
                
                # Penalty for missing big moves while flat
                if self.position == 0 and price_move >= 10.0:  # 10+ point move
                    opportunity_cost -= 0.1
                    
            # Small penalty for doing nothing when flat
            if self.position == 0 and action == TradingAction.HOLD.value:
                opportunity_cost -= 0.01
            
        # 6. Trading cost penalty
        trading_cost = 0.0
        if trade_executed:
            trading_cost = -0.05  # Spread + commission cost
            
        # Calculate total reward with safety checks
        components = [pnl_component, sharpe_component, risk_penalty, trade_quality_bonus, opportunity_cost, trading_cost]
        
        # Check all components are finite
        for i, comp in enumerate(components):
            if not np.isfinite(comp):
                if self.verbose:
                    comp_names = ['pnl', 'sharpe', 'risk', 'quality', 'opp_cost', 'trading_cost']
                    print(f"WARNING: Invalid {comp_names[i]} component: {comp}")
                components[i] = 0.0
        
        total_reward = sum(components)
        
        # Final safety check
        if not np.isfinite(total_reward):
            if self.verbose:
                print(f"WARNING: Total reward is not finite: {total_reward}, components: {components}")
            total_reward = 0.0
        
        # Ensure reward stays in reasonable range [-2, +2]
        total_reward = np.clip(total_reward, -2.0, 2.0)
        
        # Debug printing
        if self.verbose:
            print(f"PnL: {pnl_component:.3f}, Sharpe: {sharpe_component:.3f}, Risk: {risk_penalty:.3f}, "
                  f"Quality: {trade_quality_bonus:.3f}, OpCost: {opportunity_cost:.3f}, Total: {total_reward:.3f}")
        
        # Update tracking variables
        self.last_price = current_price
        if position_closed or (self.position != 0):
            self.episode_returns.append(pnl_component)
            
        return total_reward
    
    def _calculate_simple_reward(self, current_price: float, position_closed: bool, action: int = 0, trade_executed: bool = False) -> float:
        """Quality-focused reward system - prioritize winning trades over quantity"""
        
        if position_closed and len(self.episode_trades) > 0:
            # Get the last trade P&L
            last_trade = self.episode_trades[-1]
            trade_pnl = last_trade['dollar_pnl']
            
            # Base reward from P&L (normalized to [-1, 1])
            normalized_pnl = np.clip(trade_pnl / 750, -1, 1)
            
            # QUALITY BONUSES - make winning trades much more attractive than losing trades
            if trade_pnl > 0:
                # Big bonus for profitable trades - this is what we want!
                quality_bonus = 0.3
                # Extra bonus for hitting full target (15 points = $750)
                if trade_pnl >= 750:  # Hit full 15-point target
                    quality_bonus += 0.2
                    if self.verbose:
                        print(f"EXCELLENT: Full target hit! P&L=${trade_pnl:.2f}, Bonus={quality_bonus:.3f}")
            else:
                # Smaller penalty for losses (already penalized by negative PnL)
                quality_bonus = -0.05
                # But bigger penalty for max losses (hitting 5-point stop)
                if trade_pnl <= -250:  # Hit stop loss
                    quality_bonus -= 0.1
                    if self.verbose:
                        print(f"STOP LOSS: P&L=${trade_pnl:.2f}, Penalty={quality_bonus:.3f}")
            
            # Small completion bonus (reduced from 0.15 to encourage selectivity)
            completion_bonus = 0.01
            
            final_reward = normalized_pnl + quality_bonus + completion_bonus
            
            if self.verbose:
                print(f"Quality Reward: P&L=${trade_pnl:.2f} -> Base={normalized_pnl:.4f} + Quality={quality_bonus:.3f} + Complete={completion_bonus:.3f} = {final_reward:.4f}")
            
            return final_reward
        else:
            # Non-trading rewards - encourage patience and selectivity
            base_reward = 0.0
            
            if self.position == 0:
                # Reduced penalty for holding when flat (patience is now good!)
                if self.flat_time > 60:  # Increased from 30 to 60 - more patience
                    base_reward = -0.003  # Much smaller penalty than before
                else:
                    base_reward = -0.001  # Tiny penalty for normal holding
                    
                # Smaller bonus for trade attempts (quality over quantity)
                if action in [1, 2]:  # BUY or SELL attempt
                    if self.daily_trades < self.max_daily_trades and self.daily_pnl > -self.daily_loss_limit:
                        base_reward += 0.005  # Reduced from 0.02 - less reward for just trying
            
            # Small bonus for maintaining position
            elif self.position != 0:
                base_reward = 0.002  # Reduced from 0.005 - position holding is neutral
                
            if self.verbose and self.episode_step % 100 == 0:
                print(f"Quality Base: Base={base_reward:.4f}, Position={self.position}, FlatTime={self.flat_time}, Action={action}")
            
            return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Get current price data
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['Close']
        current_date = current_row['timestamp'].date()
        
        # Track action distribution and diversity
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.actions_used_this_episode.add(action)
        
        # Track consecutive same actions
        if action == self.last_action:
            self.consecutive_same_actions += 1
        else:
            self.consecutive_same_actions = 1
            
        self.last_action = action
        
        # Check for new trading day
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_pnl = 0.0
            self.daily_trades = 0
        
        # Track position closure
        position_closed = False
        close_reason = ""
        trade_executed = False
        
        # Check stop loss / profit target for existing position
        if self.position != 0:
            exit_reward = self._check_position_exits(current_price)
            if exit_reward != 0.0:  # Position was closed
                position_closed = True
                # Get close reason from the last trade
                if len(self.episode_trades) > 0:
                    close_reason = self.episode_trades[-1]['reason']
        
        # Track invalid actions for penalty calculation
        invalid_action_penalty = 0.0
        
        # Execute trading action with improved logic
        if action == TradingAction.BUY.value:
            if self.position == 0:  # Can only buy when flat
                if self.daily_trades < self.max_daily_trades and self.daily_pnl > -self.daily_loss_limit:
                    if self.verbose:
                        print(f"Step {self.current_step}: EXECUTING BUY at {current_price:.2f} (Trade #{self.daily_trades+1})")
                    self._execute_trade(action, current_price)
                    trade_executed = True
                else:
                    if self.verbose:
                        print(f"Step {self.current_step}: BLOCKED BUY - Daily trades: {self.daily_trades}/{self.max_daily_trades}, PnL: {self.daily_pnl:.2f}")
                    invalid_action_penalty = -1.0
            else:
                # Penalty for trying to buy when already positioned
                if self.verbose:
                    current_pos = "LONG" if self.position == 1 else "SHORT"
                    print(f"Step {self.current_step}: IGNORED BUY - Already in {current_pos} position")
                invalid_action_penalty = -1.0
                
        elif action == TradingAction.SELL.value:
            if self.position == 1:  # Close long position
                if self.verbose:
                    print(f"Step {self.current_step}: CLOSING LONG POSITION at {current_price:.2f}")
                self._close_position(current_price, "manual_close")
                position_closed = True
                trade_executed = True
            elif self.position == 0:  # Open short position  
                if self.daily_trades < self.max_daily_trades and self.daily_pnl > -self.daily_loss_limit:
                    if self.verbose:
                        print(f"Step {self.current_step}: EXECUTING SELL at {current_price:.2f} (Trade #{self.daily_trades+1})")
                    self._execute_trade(action, current_price)
                    trade_executed = True
                else:
                    if self.verbose:
                        print(f"Step {self.current_step}: BLOCKED SELL - Daily trades: {self.daily_trades}/{self.max_daily_trades}, PnL: {self.daily_pnl:.2f}")
                    invalid_action_penalty = -1.0
            else:  # position == -1, trying to sell when already short
                if self.verbose:
                    print(f"Step {self.current_step}: IGNORED SELL - Already in SHORT position")
                invalid_action_penalty = -0.5  # Smaller penalty since SELL could close shorts in future
        
        # Debug: Track action distribution every 100 steps  
        if self.episode_step % 100 == 0 and self.verbose:
            print(f"Episode step {self.episode_step}: Action={action} ({'HOLD' if action==0 else 'BUY' if action==1 else 'SELL'}), Position={self.position}, Daily trades={self.daily_trades}")
            
        # CRITICAL: Emergency stop if episode step gets too high (indicates termination bug)
        if self.episode_step >= 1000:
            if self.verbose:
                print(f"EMERGENCY STOP: Episode step {self.episode_step} exceeds maximum, forcing termination")
            return self._get_observation(), -10.0, True, False, self._get_info()
        
        # Update trailing stop for long positions
        if self.position == 1:
            self._update_trailing_stop(current_price)
        
        # Calculate reward (simple or advanced based on training mode)
        if self.simple_reward:
            reward = self._calculate_simple_reward(current_price, position_closed, action, trade_executed)
        else:
            reward = self._calculate_advanced_reward(
                current_price, action, trade_executed, position_closed, close_reason
            )
        
        # Calculate action diversity penalties/bonuses (DISABLED for evaluation debugging)
        diversity_penalty = 0.0
        
        # TEMPORARILY DISABLED: Diversity penalties may be breaking evaluation
        if False:  # self.episode_step < 1000:  # Disable diversity penalties
            # Penalize repetitive actions
            if self.consecutive_same_actions > 3:
                diversity_penalty -= 0.01 * (self.consecutive_same_actions - 3)
                
            # Bonus for using all action types in episode
            if len(self.actions_used_this_episode) == 3 and self.episode_step > 30:
                diversity_penalty += 0.05
                
            # Heavily penalize episodes dominated by single action type
            if self.episode_step > 50:
                total_actions = sum(self.action_counts.values())
                if total_actions > 0:
                    max_action_pct = max(self.action_counts.values()) / total_actions
                    if max_action_pct > 0.8:  # More than 80% same action
                        diversity_penalty -= 0.1
        
        # Minimal debug output for troubleshooting
        if self.verbose and self.episode_step % 100 == 0:
            print(f"Step {self.episode_step}: Action={action}, Reward={reward:.4f}, Position={self.position}")
        
        # Add penalties to reward
        reward += invalid_action_penalty + diversity_penalty
        
        # Step forward
        self.current_step += 1
        self.episode_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= len(self.df) - 1
        self.done = terminated or truncated
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward * self.reward_scaling, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get feature data for lookback window
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1
        
        feature_data = self.df.iloc[start_idx:end_idx][self.feature_cols].values
        
        # Pad if necessary
        if len(feature_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(feature_data), len(self.feature_cols)))
            feature_data = np.vstack([padding, feature_data])
        
        # Flatten feature data
        features = feature_data.flatten()
        
        # Add position and account information
        current_price = self.df.iloc[self.current_step]['Close']
        position_info = np.array([
            self.position,  # Current position (-1, 0, 1)
            self.daily_trades / self.max_daily_trades,  # Daily trade ratio
            self.daily_pnl / self.daily_loss_limit,  # Daily PnL ratio
            self.balance / self.initial_balance,  # Balance ratio
            (current_price - self.entry_price) / current_price if self.position != 0 else 0,  # Unrealized PnL ratio
            self.stop_loss_price / current_price if self.position != 0 else 0,  # Stop loss ratio
            self.profit_target_price / current_price if self.position != 0 else 0,  # Target ratio
            self.trailing_stop_price / current_price if self.position == 1 else 0,  # Trailing stop ratio
            len(self.episode_trades) / 100,  # Episode trades normalized
            self.episode_pnl / 1000,  # Episode PnL normalized
        ])
        
        observation = np.concatenate([features, position_info]).astype(np.float32)
        return observation
    
    def _execute_trade(self, action: int, price: float):
        """Execute a trade"""
        if action == TradingAction.BUY.value:
            self.position = 1
            self.entry_price = price
            self.stop_loss_price = price - self.stop_loss_points
            self.profit_target_price = price + self.profit_target_points
            self.trailing_stop_price = self.stop_loss_price
        elif action == TradingAction.SELL.value:
            self.position = -1
            self.entry_price = price
            self.stop_loss_price = price + self.stop_loss_points  
            self.profit_target_price = price - self.profit_target_points
        
        self.entry_step = self.current_step
        self.daily_trades += 1
        self.position_hold_time = 0  # Reset position hold time
    
    def _check_position_exits(self, current_price: float) -> float:
        """Check and execute position exits"""
        if self.position == 0:
            return 0.0
        
        exit_triggered = False
        exit_reason = ""
        
        if self.position == 1:  # Long position
            if current_price <= self.trailing_stop_price:
                exit_triggered = True
                exit_reason = "trailing_stop"
            elif current_price >= self.profit_target_price:
                exit_triggered = True
                exit_reason = "profit_target"
        else:  # Short position
            if current_price >= self.stop_loss_price:
                exit_triggered = True
                exit_reason = "stop_loss"
            elif current_price <= self.profit_target_price:
                exit_triggered = True
                exit_reason = "profit_target"
        
        if exit_triggered:
            return self._close_position(current_price, exit_reason)
        
        return 0.0
    
    def _close_position(self, exit_price: float, reason: str) -> float:
        """Close current position and calculate PnL"""
        if self.position == 0:
            return 0.0
        
        # Calculate PnL in points
        if self.position == 1:  # Long position
            points_pnl = exit_price - self.entry_price
        else:  # Short position
            points_pnl = self.entry_price - exit_price
        
        # Convert to dollar PnL
        dollar_pnl = points_pnl * self.point_value
        
        # Update account
        self.balance += dollar_pnl
        self.daily_pnl += dollar_pnl
        self.episode_pnl += dollar_pnl
        
        # Record trade
        trade_info = {
            'entry_step': self.entry_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position': self.position,
            'points_pnl': points_pnl,
            'dollar_pnl': dollar_pnl,
            'reason': reason
        }
        self.episode_trades.append(trade_info)
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.profit_target_price = 0.0
        self.trailing_stop_price = 0.0
        
        # Return reward (PnL in points as reward)
        return points_pnl
    
    def _update_trailing_stop(self, current_price: float):
        """Update trailing stop for long positions"""
        if self.position == 1:
            new_trailing_stop = current_price - self.stop_loss_points
            self.trailing_stop_price = max(self.trailing_stop_price, new_trailing_stop)
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL in points - WITH SAFETY CHECKS"""
        if self.position == 0:
            return 0.0
        
        # Safety checks
        if not np.isfinite(current_price) or current_price <= 0:
            return 0.0
        if not np.isfinite(self.entry_price) or self.entry_price <= 0:
            return 0.0
            
        if self.position == 1:
            pnl = current_price - self.entry_price
        else:
            pnl = self.entry_price - current_price
            
        # Final safety check
        if not np.isfinite(pnl):
            return 0.0
            
        return pnl
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Debug: Log termination checks every 100 steps
        if self.episode_step % 100 == 0 and self.verbose:
            print(f"Termination check: episode_step={self.episode_step}, daily_pnl={self.daily_pnl:.2f}, balance={self.balance:.2f}")
        
        # CRITICAL FIX: Episodes must end at market close (390 minutes = 6.5 hour trading day)
        if self.episode_step >= 390:
            if self.verbose:
                print(f"Episode ending: Market close reached (episode step {self.episode_step})")
            return True
        
        # Terminate if daily loss limit exceeded (NEGATIVE PnL only)
        if self.daily_pnl <= -self.daily_loss_limit:
            if self.verbose:
                print(f"Episode ending: Daily loss limit hit ({self.daily_pnl:.2f})")
            return True
        
        # Terminate if account blown up
        if self.balance <= self.initial_balance * 0.5:
            if self.verbose:
                print(f"Episode ending: Account blown up ({self.balance:.2f})")
            return True
        
        return False
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate for current episode"""
        if not self.episode_trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.episode_trades if trade['dollar_pnl'] > 0)
        return winning_trades / len(self.episode_trades)
    
    def _get_info(self) -> Dict:
        """Get environment info"""
        current_price = self.df.iloc[self.current_step]['Close']
        
        return {
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'position': self.position,
            'current_price': current_price,
            'episode_trades': len(self.episode_trades),
            'episode_pnl': self.episode_pnl,
            'unrealized_pnl': self._calculate_unrealized_pnl(current_price)
        }


def make_env(data_path: str, **kwargs) -> ESFuturesEnv:
    """Factory function to create ES Futures environment"""
    return ESFuturesEnv(data_path, **kwargs)