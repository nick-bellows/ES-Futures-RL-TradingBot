"""
Quick test of confidence filtering approach
"""

import sys
sys.path.append('src')

import numpy as np
from models.train_rl_agent import TradingAgentTrainer
from confidence_trader import ConfidenceTrader
import os


def quick_confidence_test(model_path: str, model_type: str, threshold: float = 0.7):
    """Quick test of confidence filtering on a single model"""
    print(f"\n{'='*60}")
    print(f"QUICK CONFIDENCE TEST - {model_type}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")
    
    # Setup
    trainer = TradingAgentTrainer("data/processed/ES_features_1min.csv")
    trainer.verify_data_splits()
    
    # Create confidence trader
    confidence_trader = ConfidenceTrader(model_path, model_type, confidence_threshold=threshold)
    
    # Create evaluation environment
    eval_env = trainer.create_env(n_envs=1, normalize=False, data_split="val")
    
    # Run episodes and track metrics
    n_episodes = 5
    results = {
        'episode_rewards': [],
        'episode_trades': [],
        'episode_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
        'confidence_decisions': [],
        'trade_outcomes': []
    }
    
    print(f"Running {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        episode_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        episode_confidences = []
        
        # Track trades at start
        start_trades = info.get('episode_trades', 0)
        start_pnl = info.get('episode_pnl', 0.0)
        
        while not done and step_count < 500:
            # Get confidence-filtered action
            action, confidence = confidence_trader.trade(obs)
            episode_confidences.append(confidence)
            
            # Track actions
            action_names = ['HOLD', 'BUY', 'SELL']
            episode_actions[action_names[action]] += 1
            results['episode_actions'][action_names[action]] += 1
            
            # Execute action
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        
        # Calculate episode results
        end_trades = info.get('episode_trades', 0)
        end_pnl = info.get('episode_pnl', 0.0)
        episode_trade_count = end_trades - start_trades
        episode_trade_pnl = end_pnl - start_pnl
        
        results['episode_rewards'].append(episode_reward)
        results['episode_trades'].append(episode_trade_count)
        results['confidence_decisions'].extend(episode_confidences)
        
        # Action distribution for episode
        total_actions = sum(episode_actions.values())
        action_pcts = {k: (v/total_actions*100) if total_actions > 0 else 0 
                      for k, v in episode_actions.items()}
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Trades={episode_trade_count}, "
              f"Trade P&L=${episode_trade_pnl:.2f}")
        print(f"    Actions: HOLD={action_pcts['HOLD']:.0f}% BUY={action_pcts['BUY']:.0f}% SELL={action_pcts['SELL']:.0f}%")
        print(f"    Avg Confidence: {np.mean(episode_confidences):.3f}")
    
    # Calculate final metrics
    total_actions = sum(results['episode_actions'].values())
    overall_action_pcts = {k: (v/total_actions*100) if total_actions > 0 else 0 
                          for k, v in results['episode_actions'].items()}
    
    confidence_stats = confidence_trader.get_statistics()
    
    print(f"\n{'-'*40}")
    print("SUMMARY RESULTS:")
    print(f"{'-'*40}")
    print(f"Mean Episode Reward:  {np.mean(results['episode_rewards']):.2f}")
    print(f"Total Trades:         {sum(results['episode_trades'])}")
    print(f"Avg Trades/Episode:   {np.mean(results['episode_trades']):.1f}")
    print(f"Action Distribution:  HOLD={overall_action_pcts['HOLD']:.0f}% "
          f"BUY={overall_action_pcts['BUY']:.0f}% SELL={overall_action_pcts['SELL']:.0f}%")
    print(f"Avg Confidence:       {np.mean(results['confidence_decisions']):.3f}")
    print(f"Confident Decisions:  {confidence_stats['confident_rate']:.1%}")
    print(f"Forced Holds:         {confidence_stats['forced_hold_rate']:.1%}")
    
    # Analysis
    trade_rate = (overall_action_pcts['BUY'] + overall_action_pcts['SELL']) / 100
    print(f"\nANALYSIS:")
    print(f"Trading Activity:     {trade_rate:.1%}")
    print(f"Confidence Threshold: {threshold}")
    
    if trade_rate > 0.3:
        print("✅ Good trading activity level")
    elif trade_rate > 0.1:
        print("⚠️ Moderate trading activity")
    else:
        print("❌ Low trading activity - may be too conservative")
        
    return results


def main():
    """Run quick confidence tests"""
    print("QUICK CONFIDENCE FILTERING TEST")
    print("="*60)
    
    # Test both models if available
    models = [
        ("models/best/ppo/ppo_best/best_model", "PPO"),
        ("models/best/dqn/dqn_best/best_model", "DQN")
    ]
    
    test_thresholds = [0.6, 0.7, 0.8]
    
    for model_path, model_type in models:
        if not os.path.exists(f"{model_path}.zip"):
            print(f"❌ {model_type} model not found at {model_path}")
            continue
            
        print(f"\nTesting {model_type} model with different thresholds:")
        
        for threshold in test_thresholds:
            try:
                results = quick_confidence_test(model_path, model_type, threshold)
                
                # Quick assessment
                trade_activity = (results['episode_actions']['BUY'] + results['episode_actions']['SELL']) / sum(results['episode_actions'].values())
                avg_reward = np.mean(results['episode_rewards'])
                total_trades = sum(results['episode_trades'])
                
                print(f"    Threshold {threshold}: Trading={trade_activity:.1%}, "
                      f"Reward={avg_reward:.1f}, Trades={total_trades}")
                
            except Exception as e:
                print(f"    Threshold {threshold}: ERROR - {e}")
    
    print(f"\n{'='*60}")
    print("QUICK TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()