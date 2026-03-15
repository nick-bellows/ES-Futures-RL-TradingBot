"""
Advanced model evaluation with multiple modes including temperature-based action selection
"""

import sys
sys.path.append('src')

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO, DQN
from models.train_rl_agent import TradingAgentTrainer
import os


class AdvancedEvaluator:
    """Advanced evaluation with temperature control and ensemble methods"""
    
    def __init__(self, data_path: str):
        self.trainer = TradingAgentTrainer(data_path)
        self.trainer.verify_data_splits()
        
    def evaluate_with_temperature(self, model, env, n_episodes: int = 10, temperature: float = 1.0, 
                                deterministic: bool = None) -> Dict:
        """
        Evaluate model with adjustable action selection temperature
        
        Args:
            temperature: 0=deterministic, 0.5=semi-random, 1.0=training distribution, >1.0=more random
            deterministic: Override for compatibility (ignored if temperature specified)
        """
        print(f"Evaluating with temperature={temperature:.1f} over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_trades = []
        episode_profits = []
        action_distributions = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_actions = {0: 0, 1: 0, 2: 0}
            step_count = 0
            done = False
            
            while not done and step_count < 500:  # Safety limit
                if temperature == 0.0 or (deterministic is True and temperature == 1.0):
                    # Pure deterministic
                    action, _ = model.predict(obs, deterministic=True)
                elif temperature == 1.0 and deterministic is not True:
                    # Use model's natural stochasticity
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    # Temperature-based sampling
                    action = self._sample_with_temperature(model, obs, temperature)
                
                # Handle numpy arrays
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if len(action.shape) > 0 else int(action)
                else:
                    action = int(action)
                
                episode_actions[action] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            episode_rewards.append(float(episode_reward))
            episode_trades.append(int(info.get('episode_trades', 0)))
            episode_profits.append(float(info.get('episode_pnl', 0)))
            action_distributions.append(episode_actions.copy())
            
            if episode % 5 == 0 or episode < 3:
                total_actions = sum(episode_actions.values())
                action_pcts = {k: (v/total_actions*100) if total_actions > 0 else 0 
                              for k, v in episode_actions.items()}
                print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Trades={episode_trades[-1]}, "
                      f"Actions: HOLD={action_pcts[0]:.0f}% BUY={action_pcts[1]:.0f}% SELL={action_pcts[2]:.0f}%")
        
        # Calculate aggregate action distribution
        total_actions = {0: 0, 1: 0, 2: 0}
        for dist in action_distributions:
            for k, v in dist.items():
                total_actions[k] += v
                
        total_count = sum(total_actions.values())
        action_percentages = {k: (v/total_count*100) if total_count > 0 else 0 
                             for k, v in total_actions.items()}
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_trades_per_episode': np.mean(episode_trades),
            'mean_profit_per_episode': np.mean(episode_profits),
            'profit_factor': (np.sum([p for p in episode_profits if p > 0]) / 
                             abs(np.sum([p for p in episode_profits if p < 0])) 
                             if any(p < 0 for p in episode_profits) else np.inf),
            'win_rate': len([p for p in episode_profits if p > 0]) / len(episode_profits) if episode_profits else 0,
            'action_distribution': action_percentages,
            'temperature': temperature,
            'episodes_completed': len(episode_rewards)
        }
        
        return results
    
    def _sample_with_temperature(self, model, obs, temperature: float) -> int:
        """Sample action using temperature-scaled logits"""
        try:
            if hasattr(model, 'policy'):
                # Get action logits/probabilities
                with torch.no_grad():
                    if hasattr(model.policy, 'predict'):
                        # DQN-style
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        if hasattr(model, 'q_net'):
                            logits = model.q_net(obs_tensor)
                        else:
                            logits = model.policy.q_net(obs_tensor)
                    else:
                        # PPO-style
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        logits = model.policy.get_distribution(obs_tensor).distribution.logits
                    
                    # Apply temperature scaling
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample from scaled distribution
                    action = torch.multinomial(probs, 1).item()
                    return action
        except Exception as e:
            print(f"Temperature sampling failed: {e}, falling back to deterministic")
            action, _ = model.predict(obs, deterministic=True)
            return int(action[0]) if isinstance(action, np.ndarray) else int(action)
    
    def evaluate_ensemble(self, model_paths: List[str], env, n_episodes: int = 10) -> Dict:
        """Evaluate using ensemble of multiple models"""
        print(f"Evaluating ensemble of {len(model_paths)} models over {n_episodes} episodes...")
        
        # Load models
        models = []
        for path in model_paths:
            try:
                if "ppo" in path.lower():
                    model = PPO.load(path)
                elif "dqn" in path.lower():
                    model = DQN.load(path)
                else:
                    print(f"Warning: Unknown model type for {path}")
                    continue
                models.append(model)
            except Exception as e:
                print(f"Failed to load model {path}: {e}")
        
        if not models:
            return {"error": "No models loaded successfully"}
        
        episode_rewards = []
        episode_trades = []
        episode_profits = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 500:
                # Get predictions from all models
                actions = []
                for model in models:
                    action, _ = model.predict(obs, deterministic=False)
                    if isinstance(action, np.ndarray):
                        action = int(action[0]) if len(action.shape) > 0 else int(action)
                    actions.append(int(action))
                
                # Use majority vote (or random choice among ties)
                action_counts = {0: 0, 1: 0, 2: 0}
                for a in actions:
                    action_counts[a] += 1
                
                max_count = max(action_counts.values())
                best_actions = [k for k, v in action_counts.items() if v == max_count]
                final_action = np.random.choice(best_actions)
                
                obs, reward, terminated, truncated, info = env.step(final_action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            episode_rewards.append(float(episode_reward))
            episode_trades.append(int(info.get('episode_trades', 0)))
            episode_profits.append(float(info.get('episode_pnl', 0)))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_trades_per_episode': np.mean(episode_trades),
            'mean_profit_per_episode': np.mean(episode_profits),
            'profit_factor': (np.sum([p for p in episode_profits if p > 0]) / 
                             abs(np.sum([p for p in episode_profits if p < 0])) 
                             if any(p < 0 for p in episode_profits) else np.inf),
            'win_rate': len([p for p in episode_profits if p > 0]) / len(episode_profits) if episode_profits else 0,
            'ensemble_size': len(models),
            'episodes_completed': len(episode_rewards)
        }


def main():
    """Run comprehensive evaluation with multiple modes"""
    print("=" * 80)
    print("ADVANCED MODEL EVALUATION")
    print("=" * 80)
    
    evaluator = AdvancedEvaluator("data/processed/ES_features_1min.csv")
    
    # Model paths to test
    ppo_path = "models/best/ppo/ppo_best/best_model"
    dqn_path = "models/best/dqn/dqn_best/best_model"
    
    # Test different evaluation modes
    evaluation_modes = [
        ("Deterministic (temp=0.0)", {"temperature": 0.0}),
        ("Conservative (temp=0.3)", {"temperature": 0.3}), 
        ("Balanced (temp=0.5)", {"temperature": 0.5}),
        ("Stochastic (temp=1.0)", {"temperature": 1.0}),
        ("Legacy Deterministic", {"deterministic": True}),
        ("Legacy Stochastic", {"deterministic": False}),
    ]
    
    for model_name, model_path in [("PPO", ppo_path), ("DQN", dqn_path)]:
        if not os.path.exists(f"{model_path}.zip"):
            print(f"\n[SKIP] {model_name} model not found at {model_path}")
            continue
            
        print(f"\n{'='*20} {model_name} EVALUATION {'='*20}")
        
        try:
            # Load model
            if "ppo" in model_path.lower():
                model = PPO.load(model_path)
            else:
                model = DQN.load(model_path)
            
            # Create evaluation environment
            eval_env = evaluator.trainer.create_env(n_envs=1, normalize=False, data_split="val")
            
            print(f"\nEvaluating {model_name} with different action selection modes:")
            print("-" * 60)
            
            results_summary = []
            
            for mode_name, kwargs in evaluation_modes:
                try:
                    print(f"\n--- {mode_name} ---")
                    results = evaluator.evaluate_with_temperature(model, eval_env, n_episodes=5, **kwargs)
                    
                    # Store for comparison
                    results_summary.append((mode_name, results))
                    
                    # Print results
                    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
                    print(f"Trades/Episode: {results['mean_trades_per_episode']:.1f}")
                    print(f"Win Rate: {results['win_rate']:.1%}")
                    print(f"Action Mix: HOLD={results['action_distribution'][0]:.0f}% "
                          f"BUY={results['action_distribution'][1]:.0f}% "
                          f"SELL={results['action_distribution'][2]:.0f}%")
                    
                except Exception as e:
                    print(f"[ERROR] {mode_name} evaluation failed: {e}")
            
            # Summary comparison
            print(f"\n{'='*20} {model_name} SUMMARY {'='*20}")
            print(f"{'Mode':<20} {'Reward':<10} {'Trades':<8} {'WinRate':<8} {'HOLD%':<6} {'Trade%':<6}")
            print("-" * 64)
            
            for mode_name, results in results_summary:
                if 'error' not in results:
                    trade_pct = results['action_distribution'][1] + results['action_distribution'][2]
                    print(f"{mode_name:<20} {results['mean_reward']:>7.2f} {results['mean_trades_per_episode']:>7.1f} "
                          f"{results['win_rate']:>7.1%} {results['action_distribution'][0]:>5.0f}% {trade_pct:>5.0f}%")
            
        except Exception as e:
            print(f"[ERROR] {model_name} evaluation failed: {e}")
    
    # Test ensemble if both models exist
    if os.path.exists(f"{ppo_path}.zip") and os.path.exists(f"{dqn_path}.zip"):
        print(f"\n{'='*20} ENSEMBLE EVALUATION {'='*20}")
        try:
            eval_env = evaluator.trainer.create_env(n_envs=1, normalize=False, data_split="val")
            ensemble_results = evaluator.evaluate_ensemble([ppo_path, dqn_path], eval_env, n_episodes=5)
            
            print(f"Ensemble Results:")
            print(f"Mean Reward: {ensemble_results['mean_reward']:.3f} ± {ensemble_results['std_reward']:.3f}")
            print(f"Trades/Episode: {ensemble_results['mean_trades_per_episode']:.1f}")
            print(f"Win Rate: {ensemble_results['win_rate']:.1%}")
            
        except Exception as e:
            print(f"[ERROR] Ensemble evaluation failed: {e}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()