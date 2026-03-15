# ES Futures Reinforcement Learning Trading Bot

An end-to-end reinforcement learning system for trading E-mini S&P 500 (ES) futures. Uses Proximal Policy Optimization (PPO) trained on 616K+ one-minute bars to generate trading signals, executed through a file-based bridge with NinjaTrader 8.

**Status**: Archived / Learning Project — see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for a full post-mortem.

---

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   DataBento     │         │   Python Bot     │         │  NinjaTrader 8  │
│   (GLBX.MDP3)  │────────>│                  │         │                 │
│   616K bars     │  train  │  PPO Model       │ signals │  ESSignalExec   │
│   47 features   │         │  370K params     │────────>│  Order Mgmt     │
│                 │         │  Feature Engine  │<────────│  ESDataBridge   │
└─────────────────┘         └──────────────────┘  data   └─────────────────┘
```

### Data Pipeline
- **Source**: DataBento API — CME Globex (`GLBX.MDP3`), 7 ES contracts (Dec 2022 - Sep 2024)
- **Features**: 47 technical indicators (price, momentum, volatility, volume, patterns)
- **Observation**: 60-bar lookback window = 2,830-dimension vectors
- **Splits**: 70/15/15 train/val/test

### Model
- **Algorithm**: PPO (Stable-Baselines3) with Gymnasium environment
- **Actions**: Hold, Buy (go long), Sell (go short / close)
- **Training**: 500K+ timesteps, checkpoints every 50K, early stopping
- **Result**: 55.8% training win rate

### Trading Rules
| Parameter | Value |
|-----------|-------|
| Position Size | 1 contract |
| Stop Loss | 5 points ($250) |
| Profit Target | 15 points ($750) |
| Risk/Reward | 1:3 |
| Max Daily Loss | $750 |
| Max Trades/Day | 5 |

---

## Project Structure

```
├── run_trading_bot.py           # Main entry point
├── core/                        # Bot runtime (NinjaTrader bot, production trader)
├── config/                      # Trading, model, and data configuration (YAML)
├── src/
│   ├── models/                  # Gymnasium RL environment, training logic
│   ├── features/                # 47-indicator feature engine
│   └── data_pipeline/           # DataBento data conversion
├── integrations/
│   ├── ninjatrader_bridge/      # File-based bridge + NinjaScript C# components
│   └── quantconnect/            # QuantConnect LEAN deployment (alternate)
├── training/                    # Model training scripts (PPO, DQN)
├── backtesting/                 # Market replay testing suite
├── tests/                       # Integration and unit tests
├── utils/                       # Contract roller, bridge setup
├── scripts/                     # Environment setup, batch files
├── docs/                        # Setup, deployment, reward system, safety
├── models/                      # Trained models and scalers (gitignored)
├── data/                        # Market data and bridge files (gitignored)
└── logs/                        # Training and trading logs (gitignored)
```

---

## Quick Start

```bash
# Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
cp .env.template .env  # add your API keys

# Train
python training/train_full.py

# Run (requires NinjaTrader 8)
python run_trading_bot.py --dry-run
```

See [docs/SETUP.md](docs/SETUP.md) for detailed environment setup and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for NinjaTrader integration.

---

## Key Technical Decisions

1. **File-based bridge** over TCP/REST — NinjaTrader's AT Interface is unreliable for real-time; file I/O proved more stable
2. **PPO over DQN** — continuous policy gradient handled the 3-action space better than discrete Q-learning
3. **47 features** aligned with QuantConnect conventions for portability across platforms
4. **Simple reward > complex reward** — the advanced reward system (Sharpe component, time decay) was built but simple PnL-based rewards trained faster

## What Went Wrong

The ML model worked. The execution layer didn't:
- Position accumulation (8 contracts despite 1-contract limit)
- Rapid signal generation outpaced order confirmation
- File bridge latency caused state desynchronization

Full analysis in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

---

## Technologies

| | |
|---|---|
| **ML/RL** | Stable-Baselines3, Gymnasium, PyTorch (CUDA) |
| **Data** | DataBento API, Pandas, NumPy, TA-Lib |
| **Execution** | NinjaTrader 8, NinjaScript (C#) |
| **Backtesting** | QuantConnect LEAN, Market Replay |
| **Monitoring** | TensorBoard, Loguru |

---

## Documentation

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — Full project post-mortem, timeline, and lessons learned
- [docs/SETUP.md](docs/SETUP.md) — Environment setup and dependency installation
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — NinjaTrader bridge deployment and backtesting
- [docs/REWARD_SYSTEM.md](docs/REWARD_SYSTEM.md) — RL reward function design
- [docs/SAFETY.md](docs/SAFETY.md) — Training and trading safety features

---

*Developed July 2024 - January 2025. Educational purposes only — not financial advice.*
