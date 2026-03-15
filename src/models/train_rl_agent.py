"""
RL Agent Training Script

Trains PPO and DQN agents on ES futures trading environment.
Implements proper hyperparameters for financial trading.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .trading_env import ESFuturesEnv


class EarlyStoppingCallback(BaseCallback):
    """
    Custom callback for early stopping when training performance plateaus
    """
    
    def __init__(self, patience: int = 20, min_improvement: float = 0.01, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.wait_count = 0
        self.stopped_training = False
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        # Check if we have evaluation results
        if hasattr(self.locals, 'infos') and len(self.locals['infos']) > 0:
            # This will be triggered by EvalCallback
            pass
            
    def update_best_reward(self, mean_reward: float) -> bool:
        """
        Update best reward and check for early stopping
        Returns True if training should continue, False if should stop
        """
        improvement = mean_reward - self.best_mean_reward
        
        if improvement > self.min_improvement:
            self.best_mean_reward = mean_reward
            self.wait_count = 0
            if self.verbose > 0:
                print(f"New best reward: {mean_reward:.4f} (improvement: {improvement:.4f})")
        else:
            self.wait_count += 1
            if self.verbose > 0:
                print(f"No improvement for {self.wait_count} evaluations (best: {self.best_mean_reward:.4f})")
                
        if self.wait_count >= self.patience:
            if self.verbose > 0:
                print(f"Early stopping triggered! No improvement for {self.patience} evaluations.")
            self.stopped_training = True
            return False
            
        return True


class CustomEvalCallback(EvalCallback):
    """
    Custom evaluation callback that works with early stopping
    """
    
    def __init__(self, *args, early_stopping_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping_callback = early_stopping_callback
        
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        # Check if we just performed an evaluation (when n_calls is divisible by eval_freq)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Get the mean reward from the last evaluation
            if len(self.evaluations_results) > 0:
                last_mean_reward = self.evaluations_results[-1]
                # Handle case where last_mean_reward might be a list/array
                if isinstance(last_mean_reward, (list, np.ndarray)):
                    last_mean_reward = float(last_mean_reward[0] if len(last_mean_reward) > 0 else 0.0)
                print(f"Evaluation result: {last_mean_reward:.4f}")
                
                # Check early stopping
                if self.early_stopping_callback is not None:
                    continue_training = self.early_stopping_callback.update_best_reward(last_mean_reward)
            
        return continue_training

# Configure device for training
def get_device():
    """Get the best available device (GPU if available, CPU otherwise)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    return device

DEVICE = get_device()

logger = logging.getLogger(__name__)


class TradingAgentTrainer:
    """Training manager for RL trading agents"""
    
    def __init__(self, 
                 data_path: str,
                 model_save_dir: str = "models/trained",
                 log_dir: str = "logs/training"):
        
        self.data_path = data_path
        self.model_save_dir = Path(model_save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint and best model directories
        self.checkpoint_dir = Path("models/checkpoints")
        self.best_model_dir = Path("models/best")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data splits (will be set by prepare_data_splits)
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def verify_data_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, int]:
        """
        Verify and display data split information
        """
        print("=" * 60)
        print("DATA SPLIT VERIFICATION")
        print("=" * 60)
        
        # Load the data
        df = pd.read_csv(self.data_path)
        total_records = len(df)
        
        # Calculate split sizes
        train_size = int(total_records * train_ratio)
        val_size = int(total_records * val_ratio)
        test_size = total_records - train_size - val_size  # Remaining records
        
        # Get date ranges
        df['timestamp'] = pd.to_datetime(df['Time'], format='%Y%m%d %H:%M')
        
        train_start = df['timestamp'].iloc[0]
        train_end = df['timestamp'].iloc[train_size-1]
        val_start = df['timestamp'].iloc[train_size]
        val_end = df['timestamp'].iloc[train_size + val_size - 1]
        test_start = df['timestamp'].iloc[train_size + val_size]
        test_end = df['timestamp'].iloc[-1]
        
        splits_info = {
            'total_records': total_records,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
        
        print(f"TOTAL RECORDS: {total_records:,}")
        print()
        print(f"TRAINING SET ({train_ratio:.0%}):")
        print(f"   Records: {train_size:,}")
        print(f"   Date Range: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(train_end - train_start).days} days")
        print()
        print(f"VALIDATION SET ({val_ratio:.0%}):")
        print(f"   Records: {val_size:,}")
        print(f"   Date Range: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(val_end - val_start).days} days")
        print()
        print(f"TEST SET ({test_ratio:.0%}):")
        print(f"   Records: {test_size:,}")
        print(f"   Date Range: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(test_end - test_start).days} days")
        print()
        print(f"Split Verification: {train_size + val_size + test_size:,} = {total_records:,} records")
        print(f"Ratios: {train_size/total_records:.1%} / {val_size/total_records:.1%} / {test_size/total_records:.1%}")
        print("=" * 60)
        print()
        
        # Actually create the data splits
        self.prepare_data_splits(df, train_ratio, val_ratio, test_ratio)
        
        return splits_info
    
    def prepare_data_splits(self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Create actual data splits for training and evaluation"""
        total_records = len(df)
        train_size = int(total_records * train_ratio)
        val_size = int(total_records * val_ratio)
        
        # Split the data chronologically (important for time series)
        self.train_data = df.iloc[:train_size].copy()
        self.val_data = df.iloc[train_size:train_size + val_size].copy()
        self.test_data = df.iloc[train_size + val_size:].copy()
        
        print("Data splits created:")
        print(f"   Training: {len(self.train_data):,} records")
        print(f"   Validation: {len(self.val_data):,} records") 
        print(f"   Test: {len(self.test_data):,} records")
        
        # CRITICAL DATA VALIDATION: Check for overlaps
        print("\nDATA VALIDATION:")
        print(f"Train: {self.train_data.index[0]} to {self.train_data.index[-1]}")
        print(f"Val: {self.val_data.index[0]} to {self.val_data.index[-1]}")
        print(f"Test: {self.test_data.index[0]} to {self.test_data.index[-1]}")
        
        # Ensure no overlap between splits
        train_end = self.train_data.index[-1]
        val_start = self.val_data.index[0]
        val_end = self.val_data.index[-1] 
        test_start = self.test_data.index[0]
        
        assert train_end < val_start, f"CRITICAL: Train/Val overlap! Train ends {train_end}, Val starts {val_start}"
        assert val_end < test_start, f"CRITICAL: Val/Test overlap! Val ends {val_end}, Test starts {test_start}"
        
        print("[OK] Data splits are non-overlapping and chronological")
        print()
        
    def create_env(self, n_envs: int = 1, normalize: bool = True, data_split: str = "train") -> gym.Env:
        """Create environment with proper data split"""
        def make_env_fn():
            # Use the appropriate data split
            if data_split == "train":
                if self.train_data is None:
                    raise ValueError("Training data not prepared. Call verify_data_splits() first.")
                data = self.train_data
            elif data_split == "val":
                if self.val_data is None:
                    raise ValueError("Validation data not prepared. Call verify_data_splits() first.")
                data = self.val_data
            elif data_split == "test":
                if self.test_data is None:
                    raise ValueError("Test data not prepared. Call verify_data_splits() first.")
                data = self.test_data
            else:
                raise ValueError(f"Invalid data_split: {data_split}. Use 'train', 'val', or 'test'.")
            
            # Data validation passed - debugging removed
            
            # Create environment with the specific data split  
            env = ESFuturesEnv(
                data=data,  # Pass DataFrame instead of file path
                verbose=False,  # Disable verbose - we know environment works
                simple_reward=True  # Use simplified reward for better learning
            )
            # Monitor wrapper - configure for evaluation mode
            if data_split in ["val", "test"]:
                # For evaluation, use minimal monitoring to avoid reward interference
                env = Monitor(env, filename=None)  # No file logging during eval
            else:
                # For training, use full monitoring with logging
                env = Monitor(env, str(self.log_dir / "monitor"))
            return env
        
        if n_envs == 1:
            env = make_env_fn()
        else:
            # Fix numpy compatibility issue
            env = make_vec_env(make_env_fn, n_envs=n_envs, seed=42)
        
        if normalize and n_envs > 1:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)  # Don't normalize rewards
        
        return env
    
    def train_ppo_agent(self, 
                       total_timesteps: int = 1000000,
                       eval_freq: int = 10000,
                       n_eval_episodes: int = 10,
                       save_best_model: bool = True,
                       checkpoint_freq: int = 50000,
                       early_stopping_patience: int = 10000,
                       min_improvement: float = 0.01) -> PPO:
        """Train PPO agent"""
        logger.info("Training PPO agent")
        
        # Create environments with proper data splits - both without normalization to avoid mismatch
        train_env = self.create_env(n_envs=4, normalize=False, data_split="train")
        eval_env = self.create_env(n_envs=1, normalize=False, data_split="val")
        
        # PPO hyperparameters optimized for trading
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,  # Stabilize value function learning
            normalize_advantage=True,
            ent_coef=0.2,  # Quadruple entropy bonus to prevent conservative convergence
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.1,  # Double target KL for more aggressive exploration
            tensorboard_log=str(self.log_dir / "ppo"),
            device=DEVICE,
            verbose=1
        )
        
        # Setup callbacks
        callbacks = []
        
        # Create PPO-specific directories
        ppo_checkpoint_dir = self.checkpoint_dir / "ppo"
        ppo_best_dir = self.best_model_dir / "ppo"
        ppo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ppo_best_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint callback - saves every 50k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(ppo_checkpoint_dir),
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            patience=early_stopping_patience,
            min_improvement=min_improvement,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        if save_best_model:
            # Custom eval callback with early stopping integration
            eval_callback = CustomEvalCallback(
                eval_env,
                best_model_save_path=str(ppo_best_dir / "ppo_best"),
                log_path=str(self.log_dir / "ppo_eval"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
                early_stopping_callback=early_stopping_callback
            )
            callbacks.append(eval_callback)
        
        # Train model
        print(f"Starting PPO training with {len(callbacks)} callbacks:")
        print(f"   - Checkpoint saving every {checkpoint_freq:,} steps")
        print(f"   - Evaluation every {eval_freq:,} steps")
        print(f"   - Early stopping patience: {early_stopping_patience} evaluations")
        print(f"   - Minimum improvement threshold: {min_improvement}")
        print()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False  # Disable to avoid tqdm dependency
        )
        
        # Check if training was stopped early
        if early_stopping_callback.stopped_training:
            print(f"Training stopped early after {early_stopping_callback.wait_count} evaluations without improvement")
            print(f"Best reward achieved: {early_stopping_callback.best_mean_reward:.4f}")
        
        # Save final model
        model.save(self.model_save_dir / "ppo_final")
        
        # Save normalization stats if using VecNormalize
        if isinstance(train_env, VecNormalize):
            train_env.save(self.model_save_dir / "ppo_vecnormalize.pkl")
        
        return model
    
    def train_dqn_agent(self,
                       total_timesteps: int = 1000000,
                       eval_freq: int = 10000,
                       n_eval_episodes: int = 10,
                       save_best_model: bool = True,
                       checkpoint_freq: int = 50000,
                       early_stopping_patience: int = 10000,
                       min_improvement: float = 0.01) -> DQN:
        """Train DQN agent"""
        logger.info("Training DQN agent")
        
        # Create environments with proper data splits (DQN doesn't support vectorized envs)
        train_env = self.create_env(n_envs=1, normalize=False, data_split="train")
        eval_env = self.create_env(n_envs=1, normalize=False, data_split="val")
        
        # DQN hyperparameters optimized for trading
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=5000,  # Reduce for faster initial learning
            batch_size=32,
            tau=0.005,  # Soft update coefficient
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10000,
            exploration_fraction=0.8,  # Extended exploration to prevent conservative convergence
            exploration_initial_eps=1.0,
            exploration_final_eps=0.25,  # Higher final exploration for continued risk-taking
            max_grad_norm=10,
            tensorboard_log=str(self.log_dir / "dqn"),
            device=DEVICE,
            verbose=1
        )
        
        # Setup callbacks
        callbacks = []
        
        # Create DQN-specific directories
        dqn_checkpoint_dir = self.checkpoint_dir / "dqn"
        dqn_best_dir = self.best_model_dir / "dqn"
        dqn_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dqn_best_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint callback - saves every 50k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(dqn_checkpoint_dir),
            name_prefix="dqn_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            patience=early_stopping_patience,
            min_improvement=min_improvement,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        if save_best_model:
            # Custom eval callback with early stopping integration
            eval_callback = CustomEvalCallback(
                eval_env,
                best_model_save_path=str(dqn_best_dir / "dqn_best"),
                log_path=str(self.log_dir / "dqn_eval"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
                early_stopping_callback=early_stopping_callback
            )
            callbacks.append(eval_callback)
        
        # Train model
        print(f"Starting DQN training with {len(callbacks)} callbacks:")
        print(f"   - Checkpoint saving every {checkpoint_freq:,} steps")
        print(f"   - Evaluation every {eval_freq:,} steps")
        print(f"   - Early stopping patience: {early_stopping_patience} evaluations")
        print(f"   - Minimum improvement threshold: {min_improvement}")
        print()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False  # Disable to avoid tqdm dependency
        )
        
        # Check if training was stopped early
        if early_stopping_callback.stopped_training:
            print(f"Training stopped early after {early_stopping_callback.wait_count} evaluations without improvement")
            print(f"Best reward achieved: {early_stopping_callback.best_mean_reward:.4f}")
        
        # Save final model
        model.save(self.model_save_dir / "dqn_final")
        
        return model
    
    def evaluate_agent(self, 
                      model_path: str, 
                      n_episodes: int = 100,
                      deterministic: bool = True) -> Dict[str, float]:
        """Evaluate trained agent"""
        logger.info(f"Evaluating agent: {model_path}")
        
        # Load model
        if "ppo" in model_path.lower():
            model = PPO.load(model_path)
        elif "dqn" in model_path.lower():
            model = DQN.load(model_path)
        else:
            raise ValueError("Unknown model type")
        
        # Create evaluation environment using test data split - with debugging
        if self.test_data is not None:
            env = self.create_env(n_envs=1, normalize=False, data_split="test")
        else:
            # Fallback to validation data if test data not available
            env = self.create_env(n_envs=1, normalize=False, data_split="val")
        
        # Extract the actual environment from VecEnv for debugging
        actual_env = env.envs[0] if hasattr(env, 'envs') else env
        
        # Run evaluation episodes
        episode_rewards = []
        episode_trades = []
        episode_profits = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 500:  # Emergency brake
                action, _ = model.predict(obs, deterministic=deterministic)
                # Handle numpy array actions
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if len(action.shape) > 0 else int(action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            if step_count >= 500:
                print(f"Episode {episode}: Hit emergency brake at {step_count} steps")
            
            # Safely extract info values with type conversion
            episode_rewards.append(float(episode_reward))
            episode_trades.append(int(info.get('episode_trades', 0)))
            episode_profits.append(float(info.get('episode_pnl', 0)))
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_trades_per_episode': np.mean(episode_trades),
            'mean_profit_per_episode': np.mean(episode_profits),
            'profit_factor': np.sum([p for p in episode_profits if p > 0]) / abs(np.sum([p for p in episode_profits if p < 0])) if any(p < 0 for p in episode_profits) else np.inf,
            'win_rate': len([p for p in episode_profits if p > 0]) / len(episode_profits) if episode_profits else 0,
        }
        
        return results
    
    def train_both_agents(self, **kwargs) -> Dict[str, Any]:
        """Train both PPO and DQN agents"""
        results = {}
        
        print("TRAINING AGENT 1/2: PPO")
        print("=" * 50)
        # Train PPO
        logger.info("Starting PPO training")
        ppo_model = self.train_ppo_agent(**kwargs)
        results['ppo'] = ppo_model
        
        print("\nTRAINING AGENT 2/2: DQN")
        print("=" * 50)
        # Train DQN  
        logger.info("Starting DQN training")
        dqn_model = self.train_dqn_agent(**kwargs)
        results['dqn'] = dqn_model
        
        print("\nBOTH AGENTS TRAINING COMPLETE!")
        print("=" * 50)
        
        return results


def main():
    """Main training script"""
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    data_path = "data/processed/ES_features_1min.csv"
    
    # Initialize trainer
    trainer = TradingAgentTrainer(data_path)
    
    # Train agents
    print("Starting RL agent training...")
    results = trainer.train_both_agents(
        total_timesteps=500000,  # Start with smaller number for testing
        eval_freq=5000,
        n_eval_episodes=5
    )
    
    print("Training complete!")
    
    # Evaluate trained models
    print("Evaluating PPO model...")
    ppo_results = trainer.evaluate_agent("models/trained/ppo_best.zip")
    print(f"PPO Results: {ppo_results}")
    
    print("Evaluating DQN model...")
    dqn_results = trainer.evaluate_agent("models/trained/dqn_best.zip")
    print(f"DQN Results: {dqn_results}")
    
    return results


if __name__ == "__main__":
    main()