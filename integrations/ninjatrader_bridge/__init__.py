"""
NinjaTrader Python Bridge for ES Futures RL Trading Bot

This package provides comprehensive integration with NinjaTrader 8 for:
- Real-time data streaming
- Order execution and management
- Market replay backtesting
- Paper trading on simulation accounts
- Performance monitoring

Components:
- nt_connector: Main NinjaTrader API connection handler
- nt_data_feed: Real-time market data streaming
- nt_order_manager: Order execution and position management
- nt_market_replay: Historical testing with Market Replay
- config: Connection settings and configuration
"""

__version__ = "1.0.0"
__author__ = "ES Futures RL Trading Bot"

from .config import NTConfig
from .nt_connector import NTConnector
from .nt_data_feed import NTDataFeed
from .nt_order_manager import NTOrderManager

__all__ = [
    'NTConfig',
    'NTConnector', 
    'NTDataFeed',
    'NTOrderManager'
]