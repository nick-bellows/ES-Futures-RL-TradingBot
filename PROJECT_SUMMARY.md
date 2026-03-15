# ES Futures RL Trading Bot - Project Summary

## Overview

A reinforcement learning trading bot for E-mini S&P 500 (ES) futures, developed from July 2024 through January 2025. The project used Proximal Policy Optimization (PPO) to learn trading signals from historical market data, with live execution through NinjaTrader 8 via a file-based bridge system.

**Final Status: Archived / Learning Project**
The ML model showed promise in training but critical execution-layer issues prevented reliable live trading.

---

## What Was Built

### Machine Learning Pipeline
- **PPO model** with 370,948 parameters trained on 616,044 one-minute OHLCV bars (Dec 2022 - Sep 2024)
- **47 technical indicators** across 6 categories: price features, moving averages, momentum, volatility, volume, and pattern recognition
- **60-bar lookback window** producing 2,830-dimension observation vectors
- **Gymnasium trading environment** with realistic constraints (stops, targets, position limits, daily loss limits)
- **Data source**: DataBento API (`GLBX.MDP3` dataset) for CME ES futures tick data
- Train/Val/Test split: 70/15/15

### Trading Strategy
| Parameter | Value |
|-----------|-------|
| Instrument | ES (E-mini S&P 500 Futures) |
| Timeframe | 1-minute bars |
| Position Size | 1 contract |
| Stop Loss | 5 points ($250) |
| Profit Target | 15 points ($750) |
| Risk/Reward | 1:3 |
| Max Daily Loss | $750 |
| Max Trades/Day | 5 |
| Confidence Threshold | 35-45% |
| Trading Hours | 9:30 AM - 4:00 PM ET |

### NinjaTrader Integration
A file-based bridge system connected the Python ML bot to NinjaTrader 8:

```
NinjaTrader (C#)          File Bridge          Python Bot
ESDataBridge.cs    -->  market_data.csv  -->  run_trading_bot.py
ESSignalExecutor.cs <--  signals.txt    <--  PPO Model Inference
```

- **ESDataBridge.cs**: NinjaScript indicator capturing real-time 1-minute OHLCV data
- **ESSignalExecutor.cs**: NinjaScript strategy reading signals and executing orders
- Signal format: `timestamp,action,quantity,confidence`

### Integration Journey
The project went through multiple platform iterations:
1. **QuantConnect** - Initial development platform, model training and backtesting
2. **Tradovate API** - Attempted live connection, blocked by CAPTCHA requirements
3. **NinjaTrader 8** - Final platform, successful file-based bridge integration

---

## Training Results

- **Training Win Rate**: 55.8%
- **Algorithm**: PPO (Stable-Baselines3)
- **Training Steps**: 500K+ timesteps with checkpoints every 50K
- **Early Stopping**: Patience-based with evaluation every 10K steps
- **GPU**: CUDA 12.1 accelerated training
- **Training Runs**: 19 PPO iterations, 11 DQN iterations logged

---

## What Worked

- PPO model successfully learned directional signals from technical indicators
- Feature engineering pipeline produced stable, normalized inputs
- NinjaTrader file bridge achieved reliable data flow in both directions
- Contract roll detection and management worked correctly
- Signal generation with confidence thresholds filtered low-quality trades
- Risk management rules (daily loss limit, trade count) functioned in the environment

## What Didn't Work (Critical Issues)

### 1. Position Size Control Failure
Despite a hard 1-contract limit, the bot accumulated up to **8 contracts** during live paper trading. The execution layer did not properly enforce position limits, likely due to rapid signal generation outpacing order confirmation.

### 2. Overtrading / Rapid Position Flips
The bot generated signals faster than NinjaTrader could process them, causing:
- Multiple entries before the first exit confirmed
- Rapid long/short flips within seconds
- Signal queue buildup in the file bridge

### 3. Stop Loss Calculation Bug
An initial bug calculated stop losses at 1.25 points instead of the intended 5 points, causing premature exits. This was identified and fixed, but demonstrated the fragility of the execution pipeline.

### 4. Latency in File Bridge
The file-based communication introduced variable latency between signal generation and order execution. In fast-moving markets, this caused:
- Stale signals being executed at worse prices
- Position state desynchronization between Python and NinjaTrader

### 5. Model Generalization
While achieving 55.8% win rate in training, the model showed typical RL overfitting characteristics. Performance degraded on out-of-sample data, particularly during regime changes and high-volatility events.

---

## Project Architecture

```
QC_TradingBot_v3/
├── core/                    # Live trading bot modules
├── config/                  # YAML/Python configuration
├── training/                # Model training scripts
├── backtesting/             # Replay testing suite
├── src/
│   ├── models/              # RL environment & training
│   ├── features/            # 47-indicator feature engine
│   └── data_pipeline/       # DataBento data conversion
├── integrations/
│   ├── ninjatrader_bridge/  # Python-side bridge code
│   │   └── ninjascript/     # C# NinjaScript components
│   └── quantconnect/       # QC deployment files
├── utils/                   # Contract roller, setup helpers
├── data/                    # Market data & bridge files
├── models/                  # Trained models & scalers
├── logs/                    # Training & trading logs
└── tests/                   # Integration tests
```

### Key Technologies
| Category | Technology |
|----------|-----------|
| ML/RL | Stable-Baselines3 (PPO, DQN), Gymnasium |
| Deep Learning | PyTorch (CUDA 12.1) |
| Data | DataBento API, Pandas, NumPy |
| Technical Analysis | TA-Lib, Pandas-TA |
| Live Trading | NinjaTrader 8 (NinjaScript C#) |
| Backtesting | QuantConnect LEAN |
| Monitoring | TensorBoard, Loguru |

---

## Lessons Learned

1. **Execution safety must be independent of the model**: Position limits, order validation, and risk checks should be enforced at every layer (model output, Python bot, bridge, and broker), not just one.

2. **File-based bridges have inherent latency issues**: For time-critical trading, direct API or socket connections are necessary. File I/O introduces unpredictable delays.

3. **RL models need extensive out-of-sample validation**: A 55.8% training win rate doesn't guarantee live performance. Market regime detection and adaptive confidence thresholds are essential.

4. **Start with the execution layer, not the model**: Building reliable order management and position tracking should precede ML model development. The best model is useless if execution is unreliable.

5. **Paper trading reveals issues that backtesting cannot**: The position accumulation bug only appeared during live paper trading due to real-world timing and order fill dynamics.

---

## Data Sources

- **Primary**: DataBento API - `GLBX.MDP3` (CME Globex) dataset
- **Schema**: `ohlcv-1m` (1-minute OHLCV bars)
- **Coverage**: 7 ES futures contracts (ESH3 through ESU4), Dec 2022 - Sep 2024
- **Volume**: 616,044 one-minute bars
- **Contract Roll Schedule**: Quarterly (H=March, M=June, U=Sept, Z=Dec)

---

## How to Run (For Reference)

### Setup
```bash
# Create environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys and paths
```

### Training
```bash
python training/train_full.py
```

### Paper Trading (requires NinjaTrader 8)
```bash
python run_trading_bot.py
```

### Backtesting
```bash
python backtesting/main_replay_test.py --mode standard
```

---

## Timeline

| Date | Milestone |
|------|-----------|
| Jul 2024 | Project started, initial data collection via DataBento |
| Aug 2024 | Feature engineering (47 indicators), RL environment built |
| Sep 2024 | PPO model training, 55.8% win rate achieved |
| Oct 2024 | QuantConnect integration attempted |
| Nov 2024 | Tradovate API integration (failed - CAPTCHA) |
| Dec 2024 | NinjaTrader file bridge built and tested |
| Jan 2025 | Live paper trading revealed critical execution issues |
| Jan 2025 | Project archived after position control failures |

---

## Disclaimer

This project is provided for **educational and research purposes only**. It is not financial advice. Trading futures involves substantial risk of loss and is not suitable for all investors. The model and code in this repository should not be used for live trading without extensive additional development, testing, and risk management.

---

*Developed July 2024 - January 2025*
