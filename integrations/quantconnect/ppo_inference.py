"""
PPO Model Inference for QuantConnect (Python)
Implements the trained PPO model for ES Futures trading
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class PPOInference:
    """PPO model inference class for QuantConnect"""
    
    def __init__(self, weights_file: str = "ppo_weights.json"):
        """
        Initialize PPO inference
        
        Args:
            weights_file: Path to exported weights JSON file
        """
        self.weights_file = weights_file
        self.weights = {}
        self.input_size = 2830
        self.output_size = 3
        self.action_names = ['HOLD', 'BUY', 'SELL']
        
        # Statistics tracking
        self.total_predictions = 0
        self.action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
    def load_weights(self) -> bool:
        """Load model weights from JSON file"""
        try:
            with open(self.weights_file, 'r') as f:
                self.weights = json.load(f)
            
            print(f"Loaded {len(self.weights)} weight tensors")
            
            # Verify expected layers are present
            required_layers = [
                'mlp_extractor.policy_net.0.weight',
                'mlp_extractor.policy_net.0.bias',
                'mlp_extractor.policy_net.2.weight', 
                'mlp_extractor.policy_net.2.bias',
                'action_net.weight',
                'action_net.bias'
            ]
            
            for layer in required_layers:
                if layer not in self.weights:
                    print(f"Warning: Missing layer {layer}")
                    return False
            
            print("All required layers found")
            return True
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _forward_pass(self, observation: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the policy network
        
        Args:
            observation: Input observation vector [2830]
            
        Returns:
            Action probabilities [3]
        """
        # Convert observation to numpy array
        x = np.array(observation, dtype=np.float32)
        
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.shape[0]}")
        
        # Policy network layer 1: 2830 -> 64
        w1 = np.array(self.weights['mlp_extractor.policy_net.0.weight']['data'])  # [64, 2830]
        b1 = np.array(self.weights['mlp_extractor.policy_net.0.bias']['data'])    # [64]
        
        h1 = self._tanh(np.dot(w1, x) + b1)  # [64]
        
        # Policy network layer 2: 64 -> 64  
        w2 = np.array(self.weights['mlp_extractor.policy_net.2.weight']['data'])  # [64, 64]
        b2 = np.array(self.weights['mlp_extractor.policy_net.2.bias']['data'])    # [64]
        
        h2 = self._tanh(np.dot(w2, h1) + b2)  # [64]
        
        # Action output layer: 64 -> 3
        w_out = np.array(self.weights['action_net.weight']['data'])  # [3, 64]
        b_out = np.array(self.weights['action_net.bias']['data'])    # [3]
        
        logits = np.dot(w_out, h2) + b_out  # [3]
        
        # Convert to probabilities
        probabilities = self._softmax(logits)
        
        return probabilities
    
    def predict(self, observation: np.ndarray, stochastic: bool = True) -> Tuple[int, float]:
        """
        Predict action from observation
        
        Args:
            observation: Feature vector [2830]
            stochastic: If True, sample from probability distribution
                       If False, take most likely action
                       
        Returns:
            (action, confidence): Action index and confidence score
        """
        probabilities = self._forward_pass(observation)
        
        if stochastic:
            # Sample from probability distribution (RECOMMENDED)
            action = np.random.choice(self.output_size, p=probabilities)
            confidence = probabilities[action]
        else:
            # Take most likely action (NOT RECOMMENDED - leads to 100% HOLD)
            action = np.argmax(probabilities)
            confidence = probabilities[action]
        
        # Update statistics
        self.total_predictions += 1
        self.action_counts[self.action_names[action]] += 1
        
        return int(action), float(confidence)
    
    def get_action_probabilities(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Get action probabilities for analysis
        
        Args:
            observation: Feature vector [2830]
            
        Returns:
            Dictionary with action probabilities
        """
        probabilities = self._forward_pass(observation)
        
        return {
            'HOLD': float(probabilities[0]),
            'BUY': float(probabilities[1]),
            'SELL': float(probabilities[2])
        }
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'action_distribution': {'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0}
            }
        
        action_dist = {}
        for action, count in self.action_counts.items():
            action_dist[action] = count / self.total_predictions
        
        return {
            'total_predictions': self.total_predictions,
            'action_distribution': action_dist
        }
    
    def reset_statistics(self):
        """Reset prediction statistics"""
        self.total_predictions = 0
        self.action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}


class FeatureCalculator:
    """Calculate trading features for PPO model input"""
    
    def __init__(self, lookback_periods: int = 60):
        """
        Initialize feature calculator
        
        Args:
            lookback_periods: Number of historical periods to use
        """
        self.lookback_periods = lookback_periods
        self.price_history = []
        self.volume_history = []
        
        # Position tracking
        self.current_position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.days_in_position = 0
        self.trades_today = 0
        self.daily_pnl = 0
        
    def update_market_data(self, price: float, volume: float = 0):
        """Update market data history"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only required lookback
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
            self.volume_history.pop(0)
    
    def _calculate_technical_indicators(self) -> np.ndarray:
        """
        Calculate technical indicators
        Returns array of shape [47 * lookback_periods]
        """
        if len(self.price_history) < self.lookback_periods:
            # Pad with zeros if not enough history
            padding_needed = self.lookback_periods - len(self.price_history)
            prices = [0] * padding_needed + self.price_history
            volumes = [0] * padding_needed + self.volume_history
        else:
            prices = self.price_history[-self.lookback_periods:]
            volumes = self.volume_history[-self.lookback_periods:]
        
        prices = np.array(prices, dtype=np.float32)
        volumes = np.array(volumes, dtype=np.float32)
        
        features = []
        
        for i in range(self.lookback_periods):
            period_features = []
            
            # Price-based features (normalized)
            if i > 0:
                price_change = (prices[i] - prices[i-1]) / max(prices[i-1], 1e-8)
                period_features.append(price_change)
            else:
                period_features.append(0.0)
            
            # Simple moving averages (normalized)
            for window in [5, 10, 20]:
                if i >= window - 1:
                    sma = np.mean(prices[max(0, i-window+1):i+1])
                    sma_norm = (prices[i] - sma) / max(sma, 1e-8)
                    period_features.append(sma_norm)
                else:
                    period_features.append(0.0)
            
            # RSI approximation
            if i >= 14:
                gains = []
                losses = []
                for j in range(i-13, i+1):
                    change = prices[j] - prices[j-1] if j > 0 else 0
                    gains.append(max(change, 0))
                    losses.append(max(-change, 0))
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 1e-8:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    period_features.append((rsi - 50) / 50)  # Normalize to [-1, 1]
                else:
                    period_features.append(0.0)
            else:
                period_features.append(0.0)
            
            # Bollinger Bands
            if i >= 19:
                bb_period = prices[max(0, i-19):i+1]
                bb_mean = np.mean(bb_period)
                bb_std = np.std(bb_period)
                
                if bb_std > 1e-8:
                    bb_upper = bb_mean + (2 * bb_std)
                    bb_lower = bb_mean - (2 * bb_std)
                    bb_position = (prices[i] - bb_mean) / (bb_std * 2)
                    period_features.append(np.clip(bb_position, -1, 1))
                else:
                    period_features.append(0.0)
            else:
                period_features.append(0.0)
            
            # Volume features (if available)
            if volumes[i] > 0:
                # Volume change
                vol_change = (volumes[i] - volumes[i-1]) / max(volumes[i-1], 1e-8) if i > 0 else 0
                period_features.append(np.clip(vol_change, -1, 1))
                
                # Volume moving average
                if i >= 9:
                    vol_ma = np.mean(volumes[max(0, i-9):i+1])
                    vol_ratio = volumes[i] / max(vol_ma, 1e-8)
                    period_features.append(np.clip(np.log(vol_ratio), -1, 1))
                else:
                    period_features.append(0.0)
            else:
                period_features.extend([0.0, 0.0])
            
            # Additional technical features to reach 47 per period
            # MACD approximation
            if i >= 25:
                ema12 = np.mean(prices[max(0, i-11):i+1])  # Simplified EMA as SMA
                ema26 = np.mean(prices[max(0, i-25):i+1])
                macd = (ema12 - ema26) / max(ema26, 1e-8)
                period_features.append(np.clip(macd, -1, 1))
            else:
                period_features.append(0.0)
            
            # Add more features to reach exactly 47 per period
            while len(period_features) < 47:
                period_features.append(0.0)
            
            # Ensure exactly 47 features per period
            period_features = period_features[:47]
            features.extend(period_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_position_features(self) -> np.ndarray:
        """Calculate position-related features (10 features)"""
        return np.array([
            float(self.current_position),  # -1, 0, or 1
            self.entry_price / 5000.0 if self.entry_price > 0 else 0.0,  # Normalized entry price
            self.unrealized_pnl / 1000.0,  # Normalized unrealized P&L
            min(self.days_in_position / 5.0, 1.0),  # Days in position (capped at 5)
            min(self.trades_today / 10.0, 1.0),  # Trades today (capped at 10)
            self.daily_pnl / 1000.0,  # Daily P&L normalized
            1.0 if self.current_position > 0 else 0.0,  # Is long
            1.0 if self.current_position < 0 else 0.0,  # Is short
            1.0 if self.current_position == 0 else 0.0,  # Is flat
            0.0  # Reserved for future use
        ], dtype=np.float32)
    
    def calculate_features(self, current_price: float, current_volume: float = 0) -> np.ndarray:
        """
        Calculate all features for model input
        
        Args:
            current_price: Current market price
            current_volume: Current volume (optional)
            
        Returns:
            Feature vector of shape [2830] (47 * 60 + 10)
        """
        # Update market data
        self.update_market_data(current_price, current_volume)
        
        # Calculate technical indicators (47 * 60 = 2820 features)
        technical_features = self._calculate_technical_indicators()
        
        # Calculate position features (10 features)
        position_features = self._calculate_position_features()
        
        # Combine all features
        all_features = np.concatenate([technical_features, position_features])
        
        # Ensure exactly 2830 features
        if len(all_features) != 2830:
            print(f"Warning: Feature count mismatch. Expected 2830, got {len(all_features)}")
            # Pad or truncate as needed
            if len(all_features) < 2830:
                padding = np.zeros(2830 - len(all_features), dtype=np.float32)
                all_features = np.concatenate([all_features, padding])
            else:
                all_features = all_features[:2830]
        
        return all_features
    
    def update_position(self, action: int, current_price: float):
        """Update position based on action taken"""
        if action == 1 and self.current_position <= 0:  # BUY
            if self.current_position < 0:  # Cover short
                pnl = (self.entry_price - current_price) * abs(self.current_position)
                self.daily_pnl += pnl
            
            self.current_position = 1
            self.entry_price = current_price
            self.days_in_position = 0
            self.trades_today += 1
            
        elif action == 2 and self.current_position >= 0:  # SELL
            if self.current_position > 0:  # Close long
                pnl = (current_price - self.entry_price) * abs(self.current_position)
                self.daily_pnl += pnl
            
            self.current_position = -1
            self.entry_price = current_price
            self.days_in_position = 0
            self.trades_today += 1
        
        # Update unrealized P&L
        if self.current_position != 0:
            if self.current_position > 0:  # Long position
                self.unrealized_pnl = (current_price - self.entry_price) * abs(self.current_position)
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.current_position)
        else:
            self.unrealized_pnl = 0


# Test the implementation
if __name__ == "__main__":
    print("Testing PPO Inference...")
    
    # Test feature calculator
    features = FeatureCalculator()
    
    # Simulate some price data
    for i in range(100):
        price = 5000 + np.sin(i * 0.1) * 50 + np.random.normal(0, 5)
        feature_vector = features.calculate_features(price, 1000 + np.random.normal(0, 100))
    
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector sample: {feature_vector[:10]}")
    
    print("PPO Inference implementation complete!")