"""
Production-ready trading system with temperature control and safety safeguards
"""

import sys
sys.path.append('src')

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from stable_baselines3 import PPO, DQN
from models.train_rl_agent import TradingAgentTrainer
from models.trading_env import ESFuturesEnv, TradingAction
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
import os


class ConfidenceLevel(Enum):
    """Confidence levels for trading decisions"""
    VERY_LOW = 0.0
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class TradingDecision:
    """Container for trading decision with metadata"""
    action: int
    confidence: float
    reasoning: str
    risk_assessment: str
    should_execute: bool


@dataclass
class SafetyLimits:
    """Production safety limits"""
    max_daily_loss: float = 750.0
    max_daily_trades: int = 5
    min_confidence_threshold: float = 0.6
    max_position_hold_time: int = 240  # 4 hours in minutes
    emergency_stop_loss: float = 1500.0  # Emergency account stop


class ProductionTrader:
    """Production-ready trader with confidence scoring and safety controls"""
    
    def __init__(self, 
                 model_path: str, 
                 temperature: float = 0.3,
                 safety_limits: Optional[SafetyLimits] = None,
                 ensemble_paths: Optional[List[str]] = None):
        """
        Initialize production trader
        
        Args:
            model_path: Path to primary trained model
            temperature: Action selection temperature (0=deterministic, 1=stochastic)
                        0.3 recommended for production (mostly deterministic with some exploration)
            safety_limits: Production safety constraints
            ensemble_paths: List of additional model paths for ensemble trading
        """
        self.temperature = temperature
        self.safety_limits = safety_limits or SafetyLimits()
        self.ensemble_paths = ensemble_paths or []
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Load models
        self.primary_model = self._load_model(model_path)
        self.ensemble_models = []
        for path in self.ensemble_paths:
            try:
                model = self._load_model(path)
                self.ensemble_models.append(model)
                self.logger.info(f"Loaded ensemble model: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to load ensemble model {path}: {e}")
        
        # Trading state
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.position_start_time = None
        self.current_position = 0
        self.account_balance = 50000.0  # Starting balance
        
        self.logger.info(f"Production trader initialized: temp={temperature}, "
                        f"ensemble_size={len(self.ensemble_models)}, "
                        f"safety_limits={safety_limits}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging"""
        logger = logging.getLogger("ProductionTrader")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        try:
            if "ppo" in model_path.lower():
                return PPO.load(model_path)
            elif "dqn" in model_path.lower():
                return DQN.load(model_path)
            else:
                # Try both
                try:
                    return PPO.load(model_path)
                except:
                    return DQN.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def get_action_with_confidence(self, observation: np.ndarray) -> TradingDecision:
        """
        Get trading action with confidence assessment
        
        Returns:
            TradingDecision with action, confidence, and execution recommendation
        """
        try:
            # Get primary model prediction
            if self.temperature == 0.0:
                primary_action, _ = self.primary_model.predict(observation, deterministic=True)
            elif self.temperature == 1.0:
                primary_action, _ = self.primary_model.predict(observation, deterministic=False)
            else:
                primary_action = self._sample_with_temperature(self.primary_model, observation, self.temperature)
            
            # Handle numpy arrays
            if isinstance(primary_action, np.ndarray):
                primary_action = int(primary_action[0]) if len(primary_action.shape) > 0 else int(primary_action)
            else:
                primary_action = int(primary_action)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(observation, primary_action)
            
            # Get ensemble agreement if available
            ensemble_agreement = 1.0  # Default if no ensemble
            if self.ensemble_models:
                ensemble_agreement = self._get_ensemble_agreement(observation, primary_action)
                confidence *= ensemble_agreement  # Adjust confidence by ensemble agreement
            
            # Generate reasoning
            reasoning = self._generate_reasoning(primary_action, confidence, ensemble_agreement)
            
            # Risk assessment
            risk_assessment = self._assess_risk(primary_action, confidence)
            
            # Safety check: should we execute this action?
            should_execute = self._safety_check(primary_action, confidence)
            
            decision = TradingDecision(
                action=primary_action,
                confidence=confidence,
                reasoning=reasoning,
                risk_assessment=risk_assessment,
                should_execute=should_execute
            )
            
            self.logger.info(f"Decision: Action={primary_action}, Confidence={confidence:.2f}, "
                           f"Execute={should_execute}, Reason={reasoning[:50]}...")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error generating trading decision: {e}")
            # Return safe default (HOLD with low confidence)
            return TradingDecision(
                action=0,  # HOLD
                confidence=0.0,
                reasoning=f"Error in decision making: {str(e)}",
                risk_assessment="HIGH RISK - System Error",
                should_execute=False
            )
    
    def _sample_with_temperature(self, model, obs: np.ndarray, temperature: float) -> int:
        """Sample action using temperature-scaled logits"""
        try:
            if hasattr(model, 'policy'):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    
                    if hasattr(model.policy, 'q_net'):
                        # DQN-style
                        logits = model.policy.q_net(obs_tensor)
                    else:
                        # PPO-style  
                        logits = model.policy.get_distribution(obs_tensor).distribution.logits
                    
                    # Apply temperature scaling
                    scaled_logits = logits / max(temperature, 0.01)  # Prevent division by zero
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample from scaled distribution
                    action = torch.multinomial(probs, 1).item()
                    return action
        except Exception as e:
            self.logger.warning(f"Temperature sampling failed: {e}, using deterministic")
            action, _ = model.predict(obs, deterministic=True)
            return int(action[0]) if isinstance(action, np.ndarray) else int(action)
    
    def _calculate_confidence(self, observation: np.ndarray, action: int) -> float:
        """Calculate confidence score for the chosen action"""
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                
                if hasattr(self.primary_model.policy, 'q_net'):
                    # DQN: Use Q-value spread as confidence
                    q_values = self.primary_model.policy.q_net(obs_tensor)
                    q_values_np = q_values.cpu().numpy()[0]
                    
                    # Confidence based on how much better the chosen action is
                    best_q = np.max(q_values_np)
                    second_best_q = np.partition(q_values_np, -2)[-2]
                    confidence = min(1.0, (best_q - second_best_q) / max(abs(best_q), 1.0))
                    
                else:
                    # PPO: Use action probability as confidence  
                    dist = self.primary_model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy()[0]
                    confidence = probs[action]
                
                return max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Medium confidence as fallback
    
    def _get_ensemble_agreement(self, observation: np.ndarray, primary_action: int) -> float:
        """Calculate ensemble agreement with primary action"""
        if not self.ensemble_models:
            return 1.0
        
        agreements = 0
        total_models = len(self.ensemble_models)
        
        for model in self.ensemble_models:
            try:
                action, _ = model.predict(observation, deterministic=(self.temperature < 0.5))
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if len(action.shape) > 0 else int(action)
                
                if action == primary_action:
                    agreements += 1
            except Exception as e:
                self.logger.warning(f"Ensemble model prediction failed: {e}")
                continue
        
        return agreements / max(total_models, 1)
    
    def _generate_reasoning(self, action: int, confidence: float, ensemble_agreement: float) -> str:
        """Generate human-readable reasoning for the decision"""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_names.get(action, "UNKNOWN")
        
        reasoning_parts = []
        
        # Action confidence
        if confidence > 0.8:
            reasoning_parts.append(f"High confidence {action_name}")
        elif confidence > 0.6:
            reasoning_parts.append(f"Moderate confidence {action_name}")
        elif confidence > 0.4:
            reasoning_parts.append(f"Low confidence {action_name}")
        else:
            reasoning_parts.append(f"Very low confidence {action_name}")
        
        # Ensemble agreement
        if self.ensemble_models:
            if ensemble_agreement > 0.8:
                reasoning_parts.append("strong ensemble agreement")
            elif ensemble_agreement > 0.5:
                reasoning_parts.append("moderate ensemble agreement")
            else:
                reasoning_parts.append("weak ensemble agreement")
        
        # Trading context
        if action == 0:
            reasoning_parts.append("maintaining current position")
        elif action in [1, 2]:
            reasoning_parts.append("active trading signal")
            
        return "; ".join(reasoning_parts)
    
    def _assess_risk(self, action: int, confidence: float) -> str:
        """Assess risk level of the proposed action"""
        risk_factors = []
        
        # Confidence-based risk
        if confidence < 0.3:
            risk_factors.append("very low confidence")
        elif confidence < 0.5:
            risk_factors.append("low confidence")
        
        # Daily limits risk
        if self.daily_trades >= self.safety_limits.max_daily_trades - 1:
            risk_factors.append("near daily trade limit")
        
        if abs(self.daily_pnl) > self.safety_limits.max_daily_loss * 0.8:
            risk_factors.append("near daily loss limit")
        
        # Position duration risk
        if (self.current_position != 0 and self.position_start_time and
            (datetime.now() - self.position_start_time).seconds / 60 > self.safety_limits.max_position_hold_time * 0.8):
            risk_factors.append("long position duration")
        
        # Emergency risk
        if self.account_balance < 45000:  # 10% account drawdown
            risk_factors.append("significant account drawdown")
        
        if not risk_factors:
            return "LOW RISK"
        elif len(risk_factors) == 1:
            return f"MEDIUM RISK: {risk_factors[0]}"
        else:
            return f"HIGH RISK: {', '.join(risk_factors)}"
    
    def _safety_check(self, action: int, confidence: float) -> bool:
        """Comprehensive safety check for action execution"""
        # Confidence threshold
        if confidence < self.safety_limits.min_confidence_threshold:
            self.logger.warning(f"Action blocked: confidence {confidence:.2f} below threshold {self.safety_limits.min_confidence_threshold}")
            return False
        
        # Daily trade limit
        if action != 0 and self.daily_trades >= self.safety_limits.max_daily_trades:
            self.logger.warning(f"Action blocked: daily trade limit reached ({self.daily_trades}/{self.safety_limits.max_daily_trades})")
            return False
        
        # Daily loss limit
        if abs(self.daily_pnl) >= self.safety_limits.max_daily_loss:
            self.logger.warning(f"Action blocked: daily loss limit exceeded (${self.daily_pnl:.2f})")
            return False
        
        # Emergency account stop
        if self.account_balance <= (50000 - self.safety_limits.emergency_stop_loss):
            self.logger.error(f"Action blocked: emergency stop loss triggered (balance: ${self.account_balance:.2f})")
            return False
        
        # Maximum position hold time
        if (self.current_position != 0 and self.position_start_time and
            (datetime.now() - self.position_start_time).seconds / 60 > self.safety_limits.max_position_hold_time):
            if action == 0:  # Allow HOLD to continue
                return True
            self.logger.warning("Action blocked: maximum position hold time exceeded")
            return False
        
        return True
    
    def update_trading_state(self, executed_action: int, trade_pnl: float = 0.0):
        """Update internal trading state after action execution"""
        if executed_action != 0:  # Not HOLD
            self.daily_trades += 1
            if executed_action in [1, 2]:  # BUY or SELL
                if self.current_position == 0:  # Opening new position
                    self.position_start_time = datetime.now()
                    self.current_position = 1 if executed_action == 1 else -1
                else:  # Closing position
                    self.current_position = 0
                    self.position_start_time = None
        
        # Update P&L and balance
        if trade_pnl != 0:
            self.daily_pnl += trade_pnl
            self.account_balance += trade_pnl
        
        self.logger.info(f"State updated: Position={self.current_position}, "
                        f"DailyTrades={self.daily_trades}, DailyPnL=${self.daily_pnl:.2f}, "
                        f"Balance=${self.account_balance:.2f}")
    
    def reset_daily_state(self):
        """Reset daily trading counters (call at start of each trading day)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.logger.info("Daily trading state reset")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status and health metrics"""
        return {
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'current_position': self.current_position,
            'position_duration_minutes': ((datetime.now() - self.position_start_time).seconds / 60 
                                        if self.position_start_time else 0),
            'safety_limits': {
                'max_daily_loss': self.safety_limits.max_daily_loss,
                'max_daily_trades': self.safety_limits.max_daily_trades,
                'min_confidence': self.safety_limits.min_confidence_threshold,
                'emergency_stop': self.safety_limits.emergency_stop_loss
            },
            'risk_status': self._assess_risk(0, 1.0),  # Overall risk assessment
            'temperature': self.temperature,
            'ensemble_size': len(self.ensemble_models)
        }


def demo_production_trader():
    """Demonstrate production trader capabilities"""
    print("=" * 80)
    print("PRODUCTION TRADER DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize production trader
        model_path = "models/best/ppo/ppo_best/best_model"
        if not os.path.exists(f"{model_path}.zip"):
            print("[ERROR] No trained model found. Run training first.")
            return
        
        # Different configurations to test
        configurations = [
            ("Conservative", 0.1, SafetyLimits(min_confidence_threshold=0.8)),
            ("Balanced", 0.3, SafetyLimits(min_confidence_threshold=0.6)),
            ("Aggressive", 0.5, SafetyLimits(min_confidence_threshold=0.4)),
        ]
        
        # Create test environment
        trainer = TradingAgentTrainer("data/processed/ES_features_1min.csv")
        trainer.verify_data_splits()
        test_env = trainer.create_env(n_envs=1, normalize=False, data_split="val")
        
        for config_name, temperature, safety_limits in configurations:
            print(f"\n{'='*20} {config_name} Configuration {'='*20}")
            print(f"Temperature: {temperature}")
            print(f"Min Confidence: {safety_limits.min_confidence_threshold}")
            
            trader = ProductionTrader(
                model_path=model_path,
                temperature=temperature,
                safety_limits=safety_limits
            )
            
            # Test trading decisions
            obs, _ = test_env.reset()
            
            print(f"\nTesting {config_name} trader for 10 decisions:")
            print("-" * 50)
            
            actions_taken = {0: 0, 1: 0, 2: 0}
            executed_actions = 0
            
            for i in range(10):
                decision = trader.get_action_with_confidence(obs)
                actions_taken[decision.action] += 1
                
                if decision.should_execute:
                    executed_actions += 1
                    # Simulate action execution
                    obs, reward, terminated, truncated, info = test_env.step(decision.action)
                    trader.update_trading_state(decision.action, 0.0)  # No P&L for demo
                else:
                    # If blocked, execute HOLD instead
                    obs, reward, terminated, truncated, info = test_env.step(0)
                
                print(f"Step {i+1:2d}: {['HOLD','BUY','SELL'][decision.action]} "
                      f"(conf={decision.confidence:.2f}) -> "
                      f"{'EXECUTED' if decision.should_execute else 'BLOCKED'}")
                
                if terminated or truncated:
                    break
            
            # Summary
            total_actions = sum(actions_taken.values())
            print(f"\n{config_name} Summary:")
            print(f"  Actions: HOLD={actions_taken[0]}, BUY={actions_taken[1]}, SELL={actions_taken[2]}")
            print(f"  Execution Rate: {executed_actions}/{total_actions} ({executed_actions/total_actions:.1%})")
            print(f"  Trading Activity: {(actions_taken[1] + actions_taken[2])/total_actions:.1%}")
            
            # Status check
            status = trader.get_trading_status()
            print(f"  Final Status: Balance=${status['account_balance']:.2f}, "
                  f"Trades={status['daily_trades']}, Risk={status['risk_status']}")
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_production_trader()