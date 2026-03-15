"""
Market Data Bridge
File-based market data reader for NinjaTrader integration
Replaces AT Interface market data subscription
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
from pathlib import Path

@dataclass
class MarketTick:
    """Market tick data structure"""
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close,
            'mid_price': self.mid_price,
            'spread': self.spread
        }

class MarketDataBridge:
    """
    Market Data Bridge for NinjaTrader file-based integration
    
    Reads real-time market data from file written by NinjaScript indicator
    Provides clean API for Python trading bot
    """
    
    def __init__(self, data_file_path: str = r"C:\NTBridge\market_data.csv"):
        self.logger = logging.getLogger(__name__)
        self.data_file_path = Path(data_file_path)
        self.status_file_path = Path(data_file_path).parent / "status.txt"
        
        # Data tracking
        self.current_tick: Optional[MarketTick] = None
        self.last_file_modified = 0
        self.tick_count = 0
        self.error_count = 0
        self.max_data_age_seconds = 2.0  # Emergency stop if data older than 2 seconds
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 0.1  # Check every 100ms
        self.last_successful_read = datetime.now()
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'read_errors': 0,
            'stale_data_warnings': 0,
            'last_price': 0.0,
            'last_update': None
        }
        
    def start_monitoring(self) -> bool:
        """Start monitoring market data file"""
        try:
            self.logger.info(f"Starting market data monitoring: {self.data_file_path}")
            
            # Check if data file exists
            if not self.data_file_path.exists():
                self.logger.error(f"Market data file does not exist: {self.data_file_path}")
                self.logger.error("Make sure ESDataBridge indicator is running in NinjaTrader")
                return False
            
            # Start monitoring thread
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Market data monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start market data monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop monitoring market data file"""
        try:
            self.logger.info("Stopping market data monitoring...")
            self.is_monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            self.logger.info("Market data monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping market data monitoring: {e}")
    
    def get_latest_tick(self) -> Optional[MarketTick]:
        """Get the latest market tick data"""
        return self.current_tick
    
    def get_current_price(self) -> float:
        """Get current market price (last or mid-price)"""
        if self.current_tick:
            return self.current_tick.last if self.current_tick.last > 0 else self.current_tick.mid_price
        return 0.0
    
    def is_data_fresh(self) -> bool:
        """Check if market data is fresh (< 2 seconds old)"""
        if not self.current_tick:
            return False
        
        age = (datetime.now() - self.current_tick.timestamp).total_seconds()
        return age < self.max_data_age_seconds
    
    def get_data_age_seconds(self) -> float:
        """Get age of current data in seconds"""
        if not self.current_tick:
            return float('inf')
        
        return (datetime.now() - self.current_tick.timestamp).total_seconds()
    
    def validate_price(self, price: float) -> bool:
        """Validate that price is reasonable for ES futures"""
        # ES typically trades between 4000-8000
        if not (4000 <= price <= 8000):
            self.logger.warning(f"Suspicious ES price: {price}")
            return False
        
        # Check for large jumps
        if self.current_tick and self.current_tick.last > 0:
            price_change = abs(price - self.current_tick.last)
            if price_change > 50:  # More than 50 point jump
                self.logger.warning(f"Large price jump: {self.current_tick.last} -> {price}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = self.stats.copy()
        stats.update({
            'is_monitoring': self.is_monitoring,
            'data_file_exists': self.data_file_path.exists(),
            'current_price': self.get_current_price(),
            'data_age_seconds': self.get_data_age_seconds(),
            'data_is_fresh': self.is_data_fresh(),
            'error_count': self.error_count
        })
        return stats
    
    def _monitor_worker(self) -> None:
        """Background worker to monitor market data file"""
        self.logger.info("Market data monitor worker started")
        
        while self.is_monitoring:
            try:
                # Check if file was modified
                if self.data_file_path.exists():
                    current_modified = self.data_file_path.stat().st_mtime
                    
                    if current_modified > self.last_file_modified:
                        self.last_file_modified = current_modified
                        self._read_market_data()
                
                # Check for stale data
                data_age = self.get_data_age_seconds()
                if data_age > self.max_data_age_seconds:
                    self.stats['stale_data_warnings'] += 1
                    if data_age > 5.0:  # Very stale
                        self.logger.error(f"CRITICAL: Market data is {data_age:.1f} seconds old!")
                    elif data_age > 2.0:
                        self.logger.warning(f"Market data is stale: {data_age:.1f} seconds old")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.error_count += 1
                self.stats['read_errors'] += 1
                self.logger.error(f"Error in monitor worker: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _read_market_data(self) -> None:
        """Read and parse market data from file"""
        try:
            # Read the single line of current market data
            with open(self.data_file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 1:
                return
            
            # Parse the latest line (should be only one line in file)
            latest_line = lines[-1].strip()
            if not latest_line or latest_line.startswith('timestamp'):
                return  # Skip header or empty lines
            
            # Parse CSV: timestamp,bid,ask,last,volume,high,low,open,close
            parts = latest_line.split(',')
            if len(parts) < 9:
                self.logger.warning(f"Invalid market data format: {latest_line}")
                return
            
            # Parse data
            timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
            bid = float(parts[1])
            ask = float(parts[2])
            last = float(parts[3])
            volume = int(parts[4])
            high = float(parts[5])
            low = float(parts[6])
            open_price = float(parts[7])
            close = float(parts[8])
            
            # Validate prices
            if not self.validate_price(last):
                return
            
            # Create market tick
            tick = MarketTick(
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                high=high,
                low=low,
                open=open_price,
                close=close
            )
            
            # Update current tick
            self.current_tick = tick
            self.tick_count += 1
            self.last_successful_read = datetime.now()
            
            # Update statistics
            self.stats['ticks_processed'] += 1
            self.stats['last_price'] = last
            self.stats['last_update'] = datetime.now()
            
            # Log periodically
            if self.tick_count % 100 == 0:
                self.logger.info(f"Market data: {last:.2f} (ticks: {self.tick_count}, age: {self.get_data_age_seconds():.1f}s)")
            
        except Exception as e:
            self.error_count += 1
            self.stats['read_errors'] += 1
            self.logger.error(f"Error reading market data: {e}")
    
    def emergency_check(self) -> Dict[str, Any]:
        """Emergency check of market data health"""
        check_result = {
            'status': 'UNKNOWN',
            'issues': [],
            'recommendations': [],
            'current_price': 0.0,
            'data_age': float('inf'),
            'file_exists': False
        }
        
        try:
            # Check if file exists
            check_result['file_exists'] = self.data_file_path.exists()
            if not check_result['file_exists']:
                check_result['status'] = 'CRITICAL'
                check_result['issues'].append('Market data file does not exist')
                check_result['recommendations'].append('Start ESDataBridge indicator in NinjaTrader')
                return check_result
            
            # Check data age
            data_age = self.get_data_age_seconds()
            check_result['data_age'] = data_age
            
            if data_age > 10:
                check_result['status'] = 'CRITICAL'
                check_result['issues'].append(f'Market data is {data_age:.1f} seconds old')
                check_result['recommendations'].append('Check if NinjaTrader is running and connected')
            elif data_age > 3:
                check_result['status'] = 'WARNING'
                check_result['issues'].append(f'Market data is {data_age:.1f} seconds old')
            else:
                check_result['status'] = 'OK'
            
            # Check current price
            current_price = self.get_current_price()
            check_result['current_price'] = current_price
            
            if current_price == 0:
                check_result['status'] = 'CRITICAL'
                check_result['issues'].append('No price data available')
            elif not self.validate_price(current_price):
                check_result['status'] = 'WARNING'
                check_result['issues'].append(f'Suspicious price: {current_price}')
            
        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f'Emergency check failed: {e}')
        
        return check_result

# Usage example and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Market Data Bridge Test")
    print("=" * 40)
    
    bridge = MarketDataBridge()
    
    # Emergency check
    print("\\nEmergency Check:")
    check = bridge.emergency_check()
    print(f"Status: {check['status']}")
    print(f"Issues: {check['issues']}")
    print(f"Current Price: {check['current_price']}")
    print(f"Data Age: {check['data_age']:.1f}s")
    
    if check['status'] == 'OK':
        # Start monitoring
        if bridge.start_monitoring():
            print("\\nMonitoring started. Press Ctrl+C to stop...")
            try:
                while True:
                    tick = bridge.get_latest_tick()
                    if tick:
                        print(f"Price: {tick.last:.2f}, Age: {bridge.get_data_age_seconds():.1f}s")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\\nStopping...")
            finally:
                bridge.stop_monitoring()
    else:
        print("\\nMarket data not available. Check NinjaTrader setup.")