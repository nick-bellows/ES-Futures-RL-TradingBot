# Safety Features

## Training Safety

### Data Split Verification
- Automatic 70/15/15 train/val/test split validation before training
- Displays date ranges and record counts for each split
- Prevents data leakage between splits

### Checkpoint Saving
- Model checkpoints saved every 50,000 timesteps
- Stored in `models/checkpoints/ppo/` and `models/checkpoints/dqn/`
- Prevents progress loss on training crashes

### Early Stopping
- Monitors mean reward improvement with patience=20 evaluation periods
- Minimum 0.01 improvement threshold required
- Automatically stops training when performance plateaus

## Trading Safety

### Risk Management Rules
| Parameter | Value |
|-----------|-------|
| Max position size | 1 contract |
| Stop loss | 5 points ($250) |
| Profit target | 15 points ($750) |
| Max daily loss | $750 |
| Max trades/day | 5 |
| Min time between trades | 60 seconds |
| Data freshness timeout | 2 seconds |

### Execution Safety
- Price validation on all signals
- Confidence threshold filtering (default: 35%)
- Position state tracking between Python and NinjaTrader
- Auto-stop on stale market data (>2 second age)
