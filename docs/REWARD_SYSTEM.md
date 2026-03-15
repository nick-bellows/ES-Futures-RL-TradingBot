# Reward System

## Overview

The trading environment (`src/models/trading_env.py`) implements an advanced reward system designed to train the PPO agent for disciplined, risk-managed trading.

## Reward Components

### 1. Scaled PnL Reward
- Normalizes realized/unrealized PnL to [-1, 1] by dividing by the profit target (15 points)
- Provides the primary learning signal

### 2. Sharpe Ratio Component (30% weight)
- Rolling calculation over last 30 returns
- Activated when sufficient history exists (>=20 returns)
- Encourages consistency over raw profit

### 3. Position Management Penalties
| Condition | Penalty |
|-----------|---------|
| Unrealized loss > $250 | -0.1 |
| Position held > 60 min | -0.001/min (time decay) |
| Daily loss > $750 | -2.0 |
| Trades > 5/day | -1.0 |

### 4. Trade Quality Bonuses
| Condition | Reward |
|-----------|--------|
| Hit 15-point profit target | +0.5 |
| Hit 5-point stop loss | -0.3 |
| Win rate > 40% | +0.5 * (win_rate - 0.4) |

### 5. Opportunity & Execution Costs
| Condition | Penalty |
|-----------|---------|
| Missed 10+ point move while flat | -0.1 |
| Holding flat (inaction) | -0.01 |
| Each trade (spread + commission) | -0.05 |

## Reward Range

All rewards are constrained to **[-2.0, +2.0]** to prevent training instability.

## Validated Results

| Scenario | Reward |
|----------|--------|
| Winning trade at target | +1.500 |
| Losing trade at stop | -0.633 |
| Daily loss limit breach | -2.000 |
| Overtrading | -1.050 |
| Inaction | -0.010/min |

## Design Rationale

The reward system creates asymmetric incentives:
- **Large positive reward** for disciplined profitable trades (+1.5)
- **Moderate penalty** for normal losses (-0.6)
- **Severe penalty** for risk violations (-2.0)

This trains the agent to prioritize risk management over aggressive trading.
