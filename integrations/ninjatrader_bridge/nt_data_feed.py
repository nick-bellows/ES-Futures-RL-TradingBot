"""
NinjaTrader Data Feed (File-based)
Real-time market data using file bridge instead of AT Interface
CRITICAL: This replaces the broken AT Interface market data subscription
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Deque
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass

from .config import NTConfig
from .market_data_bridge import MarketDataBridge, MarketTick

@dataclass
class BarData:
    """Market bar data structure matching training format"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    @classmethod
    def from_tick_data(cls, tick: MarketTick, bar_period: timedelta = timedelta(minutes=1)) -> 'BarData':
        """Create BarData from tick data (1-minute bar)"""
        # For real-time tick data, we'll use the tick as a 1-second bar
        return cls(
            timestamp=tick.timestamp,
            open=tick.last,
            high=tick.last,
            low=tick.last,
            close=tick.last,
            volume=tick.volume
        )

class NTDataFeed:
    """
    NinjaTrader Data Feed using file-based market data bridge
    
    This completely replaces the AT Interface market data subscription
    which was returning account data instead of market prices.
    """
    
    def __init__(self, config: NTConfig, connector=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Market data bridge (replaces AT Interface subscription)
        self.market_bridge = MarketDataBridge()
        
        # Data storage
        self.bars = deque(maxlen=200)  # Store last 200 bars
        self.current_contract = config.trading.instrument
        self.bar_callbacks: List[Callable] = []
        
        # State tracking
        self.streaming = False
        self.ready = False
        self.last_bar_time = datetime.min
        self.consecutive_errors = 0
        self.max_errors = 10
        
        # Statistics
        self.stats = {
            'bars_received': 0,
            'ticks_processed': 0,
            'errors': 0,
            'last_price': 0.0,
            'data_freshness_seconds': 0.0,
            'consecutive_errors': 0
        }
        
        # Bar building for 1-minute bars
        self.current_bar_data = {
            'timestamp': None,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'volume': 0,
            'tick_count': 0
        }
        
        self.logger.info(f"NTDataFeed initialized for {self.current_contract} (FILE-BASED)")
    
    def start_streaming(self, instrument: Optional[str] = None) -> bool:
        """
        Start real-time data streaming using file-based bridge
        
        Args:
            instrument: ES contract (ignored - uses file data)
            
        Returns:
            True if streaming started successfully
        """
        if self.streaming:
            self.logger.warning("Data streaming already active")
            return True
        
        try:
            self.current_contract = instrument or self.config.trading.instrument
            self.logger.info(f"Starting file-based data streaming for {self.current_contract}")
            
            # Start market data bridge
            if not self.market_bridge.start_monitoring():
                self.logger.error("Failed to start market data bridge")
                return False
            
            # Wait for initial data
            self.logger.info("Waiting for initial market data...")
            initial_wait_start = time.time()
            while time.time() - initial_wait_start < 30:  # 30 second timeout
                tick = self.market_bridge.get_latest_tick()
                if tick and self.market_bridge.is_data_fresh():
                    self.logger.info(f"Initial market data received: {tick.last:.2f}")
                    break
                time.sleep(0.5)
            else:
                self.logger.error("No initial market data received within 30 seconds")
                return False
            
            # Start streaming worker
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._streaming_worker, daemon=True)
            self.stream_thread.start()
            
            # Build initial bars from current tick
            self._initialize_bars()
            
            self.ready = True
            self.logger.info("File-based data streaming started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self) -> None:
        """Stop data streaming"""
        try:
            self.logger.info("Stopping file-based data streaming...")
            self.streaming = False
            
            # Stop market data bridge
            self.market_bridge.stop_monitoring()
            
            # Wait for thread to finish
            if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5.0)
            
            self.ready = False
            self.logger.info("Data streaming stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")
    
    def is_ready(self) -> bool:
        """Check if data feed is ready"""
        return self.ready and len(self.bars) >= 60  # Need at least 60 bars for features
    
    def get_latest_bar(self) -> Optional[BarData]:
        """Get the most recent bar"""
        return self.bars[-1] if self.bars else None
    
    def get_bars(self, count: int = 60) -> List[BarData]:
        """Get recent bars for feature calculation"""
        if not self.bars:
            return []
        
        # Return the most recent 'count' bars
        return list(self.bars)[-count:] if len(self.bars) >= count else list(self.bars)
    
    def add_bar_callback(self, callback: Callable[[BarData], None]) -> None:
        """Add callback for new bars"""
        self.bar_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data feed statistics"""
        # Get market bridge stats
        bridge_stats = self.market_bridge.get_statistics()
        
        # Combine with our stats
        combined_stats = self.stats.copy()
        combined_stats.update({
            'bridge_ticks_processed': bridge_stats.get('ticks_processed', 0),
            'bridge_read_errors': bridge_stats.get('read_errors', 0),
            'bridge_data_is_fresh': bridge_stats.get('data_is_fresh', False),
            'bridge_current_price': bridge_stats.get('current_price', 0.0),
            'bridge_data_age': bridge_stats.get('data_age_seconds', 0.0),
            'bars_stored': len(self.bars),
            'streaming': self.streaming,
            'ready': self.ready
        })
        
        return combined_stats
    
    def _streaming_worker(self) -> None:
        """Background worker to process market data ticks"""
        self.logger.info("File-based streaming worker started")
        last_tick_time = datetime.min
        
        while self.streaming:
            try:
                # Get latest tick from market bridge
                tick = self.market_bridge.get_latest_tick()
                
                if not tick:
                    time.sleep(0.1)
                    continue
                
                # Check if this is a new tick
                if tick.timestamp <= last_tick_time:
                    time.sleep(0.05)  # Short sleep if no new data
                    continue
                
                last_tick_time = tick.timestamp
                
                # Check data freshness
                data_age = self.market_bridge.get_data_age_seconds()
                self.stats['data_freshness_seconds'] = data_age
                
                if data_age > 5.0:  # Data is stale
                    self.consecutive_errors += 1
                    self.stats['consecutive_errors'] = self.consecutive_errors
                    
                    if self.consecutive_errors > self.max_errors:
                        self.logger.error("Too many consecutive data errors - stopping stream")
                        self.streaming = False
                        break
                    
                    self.logger.warning(f"Stale data detected: {data_age:.1f}s old")
                    time.sleep(1.0)
                    continue
                else:
                    self.consecutive_errors = 0
                    self.stats['consecutive_errors'] = 0
                
                # Process the tick
                self._process_tick(tick)
                
                # Update statistics
                self.stats['ticks_processed'] += 1
                self.stats['last_price'] = tick.last
                
                time.sleep(0.05)  # Small sleep to prevent excessive CPU usage
                
            except Exception as e:
                self.consecutive_errors += 1
                self.stats['errors'] += 1
                self.stats['consecutive_errors'] = self.consecutive_errors
                
                self.logger.error(f"Error in streaming worker: {e}")
                
                if self.consecutive_errors > self.max_errors:
                    self.logger.error("Too many consecutive errors - stopping stream")
                    self.streaming = False
                    break
                
                time.sleep(1.0)  # Sleep longer on error
        
        self.logger.info("Streaming worker stopped")
    
    def _process_tick(self, tick: MarketTick) -> None:
        """Process a market tick and build 1-minute bars"""
        try:
            # Round timestamp to minute boundary
            bar_time = tick.timestamp.replace(second=0, microsecond=0)
            
            # Check if we need to finalize the current bar and start a new one
            if self.current_bar_data['timestamp'] is None:
                # Initialize first bar
                self._start_new_bar(tick, bar_time)
            elif bar_time > self.current_bar_data['timestamp']:
                # Finalize current bar and start new one
                self._finalize_current_bar()
                self._start_new_bar(tick, bar_time)
            else:
                # Update current bar with tick data
                self._update_current_bar(tick)
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
    
    def _start_new_bar(self, tick: MarketTick, bar_time: datetime) -> None:
        """Start a new 1-minute bar"""
        self.current_bar_data = {
            'timestamp': bar_time,
            'open': tick.last,
            'high': tick.last,
            'low': tick.last,
            'close': tick.last,
            'volume': tick.volume,
            'tick_count': 1
        }
    
    def _update_current_bar(self, tick: MarketTick) -> None:
        """Update current bar with new tick data"""
        if self.current_bar_data['timestamp'] is None:
            return
        
        # Update OHLC
        self.current_bar_data['high'] = max(self.current_bar_data['high'], tick.last)
        self.current_bar_data['low'] = min(self.current_bar_data['low'], tick.last)
        self.current_bar_data['close'] = tick.last
        self.current_bar_data['volume'] += tick.volume
        self.current_bar_data['tick_count'] += 1
    
    def _finalize_current_bar(self) -> None:
        """Finalize current bar and add to bars collection"""
        if self.current_bar_data['timestamp'] is None:
            return
        
        try:
            # Create BarData object
            bar = BarData(
                timestamp=self.current_bar_data['timestamp'],
                open=self.current_bar_data['open'],
                high=self.current_bar_data['high'],
                low=self.current_bar_data['low'],
                close=self.current_bar_data['close'],
                volume=self.current_bar_data['volume']
            )
            
            # Add to bars collection
            self.bars.append(bar)
            self.stats['bars_received'] += 1
            
            # Log new bar occasionally
            if self.stats['bars_received'] % 10 == 0:
                self.logger.info(f"Bar {self.stats['bars_received']}: {bar.close:.2f} "
                               f"({self.current_bar_data['tick_count']} ticks)")
            
            # Notify callbacks
            for callback in self.bar_callbacks:
                try:
                    callback(bar)
                except Exception as e:
                    self.logger.error(f"Error in bar callback: {e}")
            
            self.last_bar_time = bar.timestamp
            
        except Exception as e:
            self.logger.error(f"Error finalizing bar: {e}")
    
    def _initialize_bars(self) -> None:
        """Initialize bars collection with current tick data"""
        try:
            tick = self.market_bridge.get_latest_tick()
            if not tick:
                return
            
            # Create initial bars (simulate historical data)
            now = datetime.now()
            
            for i in range(60, 0, -1):  # Create 60 bars going back in time
                bar_time = now - timedelta(minutes=i)
                bar_time = bar_time.replace(second=0, microsecond=0)
                
                # Create bar with current price (no historical data available)
                bar = BarData(
                    timestamp=bar_time,
                    open=tick.last,
                    high=tick.last,
                    low=tick.last,
                    close=tick.last,
                    volume=tick.volume
                )
                
                self.bars.append(bar)
            
            self.stats['bars_received'] = len(self.bars)
            self.logger.info(f"Initialized with {len(self.bars)} synthetic historical bars")
            
        except Exception as e:
            self.logger.error(f"Error initializing bars: {e}")

# Compatibility function for existing code
def create_data_feed(config: NTConfig, connector=None) -> NTDataFeed:
    """Create a new NTDataFeed instance"""
    return NTDataFeed(config, connector)