#!/usr/bin/env python3
"""
Timestamp Parsing Debug Script
Identifies and fixes timestamp parsing issues in market_data.csv
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config.bridge_config import MARKET_DATA_FILE

def debug_csv_timestamps():
    """Debug timestamp parsing from market_data.csv"""
    
    print("Timestamp Parsing Debug")
    print("=" * 40)
    
    data_file = Path(MARKET_DATA_FILE)
    
    if not data_file.exists():
        print(f"ERROR: Market data file not found: {data_file}")
        return
    
    print(f"Reading file: {data_file}")
    
    # Read raw lines
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        if len(lines) < 2:
            print("ERROR: File has no data rows")
            return
        
        # Show last few lines
        print("\nLast 3 raw lines:")
        for i, line in enumerate(lines[-3:], len(lines)-2):
            print(f"  {i}: {repr(line)}")
        
        # Parse last data line
        last_line = lines[-1].strip()
        print(f"\nLast data line: {repr(last_line)}")
        
        if last_line.startswith('timestamp'):
            print("ERROR: Last line is header, no data available")
            return
        
        # Split CSV
        parts = last_line.split(',')
        print(f"CSV parts: {len(parts)} columns")
        
        if len(parts) < 9:
            print(f"ERROR: Expected 9 columns, got {len(parts)}")
            return
        
        raw_timestamp = parts[0]
        print(f"Raw timestamp string: '{raw_timestamp}'")
        
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

def test_timestamp_formats(timestamp_str):
    """Test different timestamp parsing formats"""
    
    print(f"\nTesting timestamp formats for: '{timestamp_str}'")
    print("-" * 50)
    
    formats_to_try = [
        '%Y-%m-%d %H:%M:%S.%f',      # 2025-09-14 21:54:23.495000
        '%Y-%m-%d %H:%M:%S',         # 2025-09-14 21:54:23
        '%Y-%m-%d %H:%M:%S.%f',      # Padded microseconds
    ]
    
    for i, fmt in enumerate(formats_to_try, 1):
        try:
            # Test original
            parsed = datetime.strptime(timestamp_str, fmt)
            age = (datetime.now() - parsed).total_seconds()
            print(f"  Method {i}: SUCCESS - {parsed} (age: {age:.1f}s)")
            return parsed
            
        except ValueError as e:
            print(f"  Method {i}: FAILED - {e}")
    
    # Try with padding milliseconds
    if '.' in timestamp_str:
        decimal_part = timestamp_str.split('.')[-1]
        if len(decimal_part) == 3:  # 3-digit milliseconds
            padded_timestamp = timestamp_str + '000'  # Pad to 6 digits
            print(f"\nTrying padded timestamp: '{padded_timestamp}'")
            try:
                parsed = datetime.strptime(padded_timestamp, '%Y-%m-%d %H:%M:%S.%f')
                age = (datetime.now() - parsed).total_seconds()
                print(f"  Padded method: SUCCESS - {parsed} (age: {age:.1f}s)")
                return parsed
            except ValueError as e:
                print(f"  Padded method: FAILED - {e}")
    
    # Try pandas (more robust)
    print(f"\nTrying pandas.to_datetime...")
    try:
        parsed = pd.to_datetime(timestamp_str)
        # Convert to datetime object if it's pandas Timestamp
        if hasattr(parsed, 'to_pydatetime'):
            parsed = parsed.to_pydatetime()
        age = (datetime.now() - parsed).total_seconds()
        print(f"  Pandas method: SUCCESS - {parsed} (age: {age:.1f}s)")
        return parsed
    except Exception as e:
        print(f"  Pandas method: FAILED - {e}")
    
    print("  ERROR: All parsing methods failed!")
    return None

def test_current_bridge_parsing():
    """Test how the current bridge is parsing timestamps"""
    
    print("\nTesting current bridge parsing...")
    print("-" * 40)
    
    try:
        from ninjatrader_bridge.market_data_bridge import ProjectMarketDataBridge
        
        bridge = ProjectMarketDataBridge()
        
        # Force read data
        bridge._read_market_data()
        
        current_tick = bridge.get_latest_tick()
        
        if current_tick:
            print(f"Bridge parsed timestamp: {current_tick.timestamp}")
            print(f"Bridge timestamp type: {type(current_tick.timestamp)}")
            
            age = bridge.get_data_age_seconds()
            print(f"Bridge calculated age: {age} seconds")
            
            if age == float('inf'):
                print("ERROR: Bridge returning infinite age!")
            else:
                print(f"Age is finite: {age:.1f}s")
                
        else:
            print("ERROR: Bridge returned no current tick")
    
    except Exception as e:
        print(f"ERROR testing bridge: {e}")

def test_manual_age_calculation():
    """Manually test age calculation"""
    
    print("\nTesting manual age calculation...")
    print("-" * 40)
    
    data_file = Path(MARKET_DATA_FILE)
    
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        last_line = lines[-1].strip()
        if last_line.startswith('timestamp'):
            print("No data to test")
            return
        
        parts = last_line.split(',')
        raw_timestamp = parts[0]
        
        # Try to parse timestamp
        parsed_timestamp = test_timestamp_formats(raw_timestamp)
        
        if parsed_timestamp:
            now = datetime.now()
            print(f"\nManual age calculation:")
            print(f"  Parsed timestamp: {parsed_timestamp}")
            print(f"  Current time:     {now}")
            
            # Calculate difference
            time_diff = now - parsed_timestamp
            print(f"  Time difference:  {time_diff}")
            
            age_seconds = time_diff.total_seconds()
            print(f"  Age in seconds:   {age_seconds}")
            
            if age_seconds < 0:
                print("  WARNING: Negative age - timestamp is in the future!")
            elif age_seconds > 3600:
                print("  WARNING: Very old data (>1 hour)")
            else:
                print(f"  Age looks reasonable: {age_seconds:.1f} seconds")
    
    except Exception as e:
        print(f"ERROR in manual calculation: {e}")

def main():
    """Run all timestamp debugging tests"""
    
    # Test 1: Read and examine CSV
    debug_csv_timestamps()
    
    # Test 2: Test different parsing formats
    data_file = Path(MARKET_DATA_FILE)
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            last_line = lines[-1].strip()
            if not last_line.startswith('timestamp'):
                parts = last_line.split(',')
                raw_timestamp = parts[0]
                test_timestamp_formats(raw_timestamp)
    except:
        pass
    
    # Test 3: Current bridge behavior
    test_current_bridge_parsing()
    
    # Test 4: Manual age calculation
    test_manual_age_calculation()
    
    print("\n" + "=" * 40)
    print("DEBUG COMPLETE")
    print("=" * 40)

if __name__ == "__main__":
    main()