"""
Full training script using Python 3.11 with GPU support
Run with: py -3.11 train_full.py
"""

import sys
import os
sys.path.append('src')

from models.train_rl_agent import TradingAgentTrainer
import torch

def main():
    print("=" * 80)
    print("ES FUTURES RL TRADING BOT - FULL TRAINING")
    print("=" * 80)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 80)
    print()
    
    # Configuration
    data_path = "data/processed/ES_features_1min.csv"
    
    # Initialize trainer
    trainer = TradingAgentTrainer(data_path)
    
    # SAFETY FEATURE 1: Verify data splits before training
    print("SAFETY CHECK: Verifying data splits...")
    splits_info = trainer.verify_data_splits()
    
    # Ask for user confirmation
    response = input("Do you want to proceed with training? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("Training cancelled by user.")
        return None
    
    print("Starting training with enhanced safety features:")
    print("   - Checkpoint saving every 50,000 steps")
    print("   - Early stopping with patience=20 (200k steps)")
    print("   - Best model auto-saving")
    print("   - Minimum improvement threshold: 0.01")
    print()
    
    # Train both agents with safety features
    print("Starting full training (500K timesteps each)...")
    results = trainer.train_both_agents(
        total_timesteps=500000,  # Full training
        eval_freq=10000,         # Evaluate every 10k steps
        n_eval_episodes=5,       # Use 5 episodes for evaluation
        checkpoint_freq=50000,   # Save checkpoint every 50k steps
        early_stopping_patience=20,  # Stop if no improvement for 20 evals
        min_improvement=0.01     # Minimum improvement threshold
    )
    
    print("Training complete!")
    
    # Evaluate trained models
    print("EVALUATING TRAINED MODELS")
    print("=" * 50)
    
    print("Evaluating PPO model...")
    # EvalCallback saves models as: models/best/ppo/ppo_best/best_model.zip
    ppo_model_path = "models/best/ppo/ppo_best/best_model"  # EvalCallback structure
    ppo_alt_path = "models/best/ppo/ppo_best"  # Alternative structure
    
    if os.path.exists(f"{ppo_model_path}.zip"):
        try:
            ppo_results = trainer.evaluate_agent(ppo_model_path, n_episodes=20)
            print(f"[OK] PPO Results: {ppo_results}")
        except Exception as e:
            print(f"[ERROR] PPO evaluation failed: {e}")
            ppo_results = {'error': str(e)}
    elif os.path.exists(f"{ppo_alt_path}.zip"):
        try:
            ppo_results = trainer.evaluate_agent(ppo_alt_path, n_episodes=20)
            print(f"[OK] PPO Results: {ppo_results}")
        except Exception as e:
            print(f"[ERROR] PPO evaluation failed: {e}")
            ppo_results = {'error': str(e)}
    else:
        print(f"[WARNING] PPO model not found at {ppo_model_path}.zip or {ppo_alt_path}.zip, skipping evaluation")
        ppo_results = {'error': 'Model file not found'}
    
    print("\nEvaluating DQN model...")
    # EvalCallback saves models as: models/best/dqn/dqn_best/best_model.zip
    dqn_model_path = "models/best/dqn/dqn_best/best_model"  # EvalCallback structure
    dqn_alt_path = "models/best/dqn/dqn_best"  # Alternative structure
    
    if os.path.exists(f"{dqn_model_path}.zip"):
        try:
            dqn_results = trainer.evaluate_agent(dqn_model_path, n_episodes=20)
            print(f"[OK] DQN Results: {dqn_results}")
        except Exception as e:
            print(f"[ERROR] DQN evaluation failed: {e}")
            dqn_results = {'error': str(e)}
    elif os.path.exists(f"{dqn_alt_path}.zip"):
        try:
            dqn_results = trainer.evaluate_agent(dqn_alt_path, n_episodes=20)
            print(f"[OK] DQN Results: {dqn_results}")
        except Exception as e:
            print(f"[ERROR] DQN evaluation failed: {e}")
            dqn_results = {'error': str(e)}
    else:
        print(f"[WARNING] DQN model not found at {dqn_model_path}.zip or {dqn_alt_path}.zip, skipping evaluation")
        dqn_results = {'error': 'Model file not found'}
    
    # Summary
    print("\nTRAINING SUMMARY")
    print("=" * 50)
    
    if 'error' not in ppo_results:
        print(f"PPO Mean Reward: {ppo_results.get('mean_reward', 'N/A'):.4f}")
        print(f"PPO Win Rate: {ppo_results.get('win_rate', 0):.2%}")
    else:
        print("[ERROR] PPO evaluation failed")
        
    if 'error' not in dqn_results:
        print(f"DQN Mean Reward: {dqn_results.get('mean_reward', 'N/A'):.4f}")
        print(f"DQN Win Rate: {dqn_results.get('win_rate', 0):.2%}")
    else:
        print("[ERROR] DQN evaluation failed")
        
    print("\nModel Files Saved:")
    print("   - Best models: models/best/ppo/ and models/best/dqn/")
    print("   - Checkpoints: models/checkpoints/ppo/ and models/checkpoints/dqn/")
    print("   - Logs: logs/training/")
    
    print(f"\n[SUCCESS] Training completed successfully!")
    print("=" * 50)
    
    return results, ppo_results, dqn_results

if __name__ == "__main__":
    main()