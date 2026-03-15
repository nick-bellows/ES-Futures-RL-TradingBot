"""
Signal Writer for NinjaTrader Integration
Writes ML model trading signals to file for NinjaScript strategy
"""

import os
import time
import logging
import tempfile
from datetime import datetime
from typing import Optional, Union
from pathlib import Path
from enum import Enum

class TradingAction(Enum):
    """Trading action enumeration"""
    BUY = "BUY"
    SELL = "SELL" 
    FLAT = "FLAT"
    HOLD = "HOLD"

class SignalWriter:
    """
    Signal Writer for NinjaTrader file-based integration
    
    Writes trading signals from Python ML bot to file for NinjaScript consumption
    Handles atomic writes and signal validation
    """
    
    def __init__(self, signal_file_path: str = r"C:\NTBridge\signals.txt"):
        self.logger = logging.getLogger(__name__)
        self.signal_file_path = Path(signal_file_path)
        self.last_signal = ""
        self.signal_count = 0
        self.min_confidence = 0.3
        self.min_time_between_signals = 1.0  # Minimum 1 second between signals
        self.last_signal_time = datetime.min
        
        # Ensure directory exists
        self.signal_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear signal file on initialization
        self.clear_signals()
        
        self.logger.info(f"SignalWriter initialized: {self.signal_file_path}")
    
    def write_signal(self, action: Union[TradingAction, str, int], 
                    current_price: float, 
                    confidence: float,
                    quantity: int = 1) -> bool:
        """
        Write trading signal to file for NinjaScript
        
        Args:
            action: Trading action (BUY/SELL/FLAT/HOLD or numeric 1/2/0)
            current_price: Current market price
            confidence: Model confidence (0.0 to 1.0)
            quantity: Position quantity (default 1)
            
        Returns:
            True if signal written successfully
        """
        try:
            # Convert numeric action to string
            if isinstance(action, int):
                action_map = {0: TradingAction.HOLD, 1: TradingAction.BUY, 2: TradingAction.SELL}
                action = action_map.get(action, TradingAction.HOLD)
            elif isinstance(action, str):
                action = TradingAction(action.upper())
            
            # Validate inputs
            if not self._validate_signal(action, current_price, confidence):
                return False
            
            # Check minimum time between signals
            now = datetime.now()
            if (now - self.last_signal_time).total_seconds() < self.min_time_between_signals:
                self.logger.debug("Signal throttled - too soon after last signal")
                return False
            
            # Don't write HOLD signals (NinjaScript ignores them anyway)
            if action == TradingAction.HOLD:
                return True
            
            # Create signal string
            signal_content = self._format_signal(action, current_price, confidence, quantity, now)
            
            # Check if signal changed (don't write duplicate signals)
            if signal_content == self.last_signal:
                self.logger.debug("Signal unchanged - skipping write")
                return True
            
            # Write signal atomically
            if self._write_signal_atomic(signal_content):
                self.last_signal = signal_content
                self.last_signal_time = now
                self.signal_count += 1
                
                self.logger.info(f"Signal written: {action.value} @ {current_price:.2f} (conf: {confidence:.1%})")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error writing signal: {e}")
            return False
    
    def write_buy_signal(self, price: float, confidence: float, quantity: int = 1) -> bool:
        """Write BUY signal"""
        return self.write_signal(TradingAction.BUY, price, confidence, quantity)
    
    def write_sell_signal(self, price: float, confidence: float, quantity: int = 1) -> bool:
        """Write SELL signal"""
        return self.write_signal(TradingAction.SELL, price, confidence, quantity)
    
    def write_flat_signal(self, price: float, confidence: float = 1.0) -> bool:
        """Write FLAT/CLOSE signal"""
        return self.write_signal(TradingAction.FLAT, price, confidence, 0)
    
    def clear_signals(self) -> bool:
        """Clear/delete signal file"""
        try:
            if self.signal_file_path.exists():
                self.signal_file_path.unlink()
                self.logger.info("Signal file cleared")
            
            self.last_signal = ""
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing signals: {e}")
            return False
    
    def get_last_signal(self) -> Optional[str]:
        """Get the last written signal"""
        try:
            if self.signal_file_path.exists():
                with open(self.signal_file_path, 'r') as f:
                    return f.read().strip()
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading last signal: {e}")
            return None
    
    def get_signal_stats(self) -> dict:
        """Get signal writing statistics"""
        return {
            'signal_count': self.signal_count,
            'last_signal': self.last_signal,
            'last_signal_time': self.last_signal_time,
            'signal_file_exists': self.signal_file_path.exists(),
            'signal_file_path': str(self.signal_file_path)
        }
    
    def _validate_signal(self, action: TradingAction, price: float, confidence: float) -> bool:
        """Validate signal parameters"""
        # Check confidence threshold
        if confidence < self.min_confidence:
            self.logger.warning(f"Signal confidence too low: {confidence:.4f} < {self.min_confidence}")
            return False
        
        # Check price reasonableness for ES futures
        if not (4000 <= price <= 8000):
            self.logger.error(f"Invalid ES price for signal: {price}")
            return False
        
        # Check confidence range
        if not (0.0 <= confidence <= 1.0):
            self.logger.error(f"Invalid confidence range: {confidence}")
            return False
        
        return True
    
    def _format_signal(self, action: TradingAction, price: float, confidence: float, 
                      quantity: int, timestamp: datetime) -> str:
        """Format signal string for NinjaScript consumption"""
        # Format: timestamp,action,quantity,confidence
        # Example: 2024-12-10 14:30:15.123,BUY,1,0.4523
        return f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{action.value},{quantity},{confidence:.6f}"
    
    def _write_signal_atomic(self, signal_content: str) -> bool:
        """Write signal file atomically to prevent partial reads"""
        try:
            # Use temporary file in same directory for atomic write
            temp_dir = self.signal_file_path.parent
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False, suffix='.tmp') as temp_file:
                temp_file.write(signal_content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_path = temp_file.name
            
            # Atomic move (rename) to final location
            os.replace(temp_path, self.signal_file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in atomic write: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return False

# Legacy compatibility function (matches current bot interface)
def write_signal_for_ninjatrader(action: Union[int, str], price: float, confidence: float) -> bool:
    """
    Legacy compatibility function for existing bot code
    
    Args:
        action: 1=BUY, 2=SELL, 0=HOLD or string action
        price: Current price
        confidence: Model confidence
        
    Returns:
        True if signal written successfully
    """
    # Use global signal writer instance
    global _global_signal_writer
    
    if '_global_signal_writer' not in globals():
        _global_signal_writer = SignalWriter()
    
    return _global_signal_writer.write_signal(action, price, confidence)

# Test and usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Signal Writer Test")
    print("=" * 30)
    
    # Create signal writer
    writer = SignalWriter()
    
    # Test different signal types
    test_price = 6595.50
    
    print(f"\\nTesting signal writing at price {test_price}...")
    
    # Test BUY signal
    success = writer.write_buy_signal(test_price, 0.4523)
    print(f"BUY signal: {'SUCCESS' if success else 'FAILED'}")
    
    time.sleep(1.5)  # Wait for minimum interval
    
    # Test SELL signal
    success = writer.write_sell_signal(test_price - 2.5, 0.3891)
    print(f"SELL signal: {'SUCCESS' if success else 'FAILED'}")
    
    time.sleep(1.5)
    
    # Test FLAT signal
    success = writer.write_flat_signal(test_price - 1.0)
    print(f"FLAT signal: {'SUCCESS' if success else 'FAILED'}")
    
    # Show last signal
    last_signal = writer.get_last_signal()
    print(f"\\nLast signal in file: {last_signal}")
    
    # Show statistics
    stats = writer.get_signal_stats()
    print(f"\\nStatistics:")
    print(f"  Signals written: {stats['signal_count']}")
    print(f"  Last signal time: {stats['last_signal_time']}")
    print(f"  File exists: {stats['signal_file_exists']}")
    
    # Test legacy function
    print(f"\\nTesting legacy function...")
    success = write_signal_for_ninjatrader(1, test_price + 1.0, 0.4156)
    print(f"Legacy BUY signal: {'SUCCESS' if success else 'FAILED'}")
    
    print(f"\\nSignal file location: {writer.signal_file_path}")
    print("Check NinjaTrader ESSignalExecutor strategy for execution.")