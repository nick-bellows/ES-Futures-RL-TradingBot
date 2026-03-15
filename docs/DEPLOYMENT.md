# NinjaTrader Deployment Guide

## Architecture

The system uses a file-based bridge for Python-NinjaTrader communication:

```
NinjaTrader 8 (C#)              Python Bot
ESDataBridge.cs     ──────>     market_data.csv     ──────>  run_trading_bot.py
ESSignalExecutor.cs <──────     signals.txt         <──────  PPO Model Inference
```

### Bridge Files
```
data/bridge/
├── market_data.csv     # Real-time OHLCV from NinjaScript (NT -> Python)
├── signals.txt         # Trading signals (Python -> NT)
├── status.txt          # NinjaScript status updates
└── execution_log.txt   # Order execution log
```

## Step 1: Initialize Bridge

```bash
python utils/setup_bridge.py
python utils/setup_bridge.py --status  # verify
```

## Step 2: Deploy NinjaScript Components

Copy the C# files to NinjaTrader's custom script directories:

| Source | Target |
|--------|--------|
| `integrations/ninjatrader_bridge/ninjascript/UpdatedESDataBridge.cs` | `Documents/NinjaTrader 8/bin/Custom/Indicators/` |
| `integrations/ninjatrader_bridge/ninjascript/UpdatedESSignalExecutor.cs` | `Documents/NinjaTrader 8/bin/Custom/Strategies/` |

Then open NinjaScript Editor and press **F5** to compile.

## Step 3: Chart Configuration

### ESDataBridge Indicator
1. Open ES futures chart (1-minute timeframe)
2. Right-click -> Indicators -> Add **ESDataBridge**
3. Verify status text: "ES Data Bridge: X ticks"

### ESSignalExecutor Strategy
1. Same chart -> Right-click -> Strategies -> Add **ESSignalExecutor**
2. Configure: Stop Loss = 5.0, Profit Target = 15.0, Max Position = 1

## Step 4: Run the Bot

```bash
python run_trading_bot.py                    # production mode
python run_trading_bot.py --dry-run          # no signals sent
python run_trading_bot.py --check-only       # verify prerequisites
```

## Step 5: Integration Test

```bash
python tests/test_bridge_integration.py
```

## Backtesting with Market Replay

```bash
python backtesting/main_replay_test.py --mode standard
python backtesting/main_replay_test.py --mode all --quick
```

### Test Types
- **Standard**: 10 market conditions (trending, range-bound, high-vol)
- **Stress**: Crash days, geopolitical events, earnings
- **Sensitivity**: Confidence threshold optimization (0.25-0.50)

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| No market data | Check ESDataBridge indicator is active on ES chart |
| Signals not executing | Verify ESSignalExecutor strategy is enabled |
| File permission errors | Run `python utils/setup_bridge.py` as admin |
| Stale data (>2s age) | System auto-stops; restart NinjaTrader indicator |
