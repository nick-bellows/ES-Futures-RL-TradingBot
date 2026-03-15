"""
Real-time Data Flow Test
Monitors market_data.csv for continuous updates from NinjaScript ESDataBridge
"""

import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.bridge_config import MARKET_DATA_FILE

def test_data_flow(duration_seconds=10):
    """
    Monitor market_data.csv for continuous updates
    
    Args:
        duration_seconds: How long to monitor the file
    """
    print("Real-time Data Flow Test")
    print("=" * 40)
    print(f"Monitoring: {MARKET_DATA_FILE}")
    print(f"Duration: {duration_seconds} seconds")
    print()
    
    data_file = Path(MARKET_DATA_FILE)
    
    # Check if file exists
    if not data_file.exists():
        print("❌ ERROR: Market data file does not exist!")
        print("   Make sure NinjaScript ESDataBridge is running")
        return False
    
    # Get initial state
    initial_size = data_file.stat().st_size
    initial_mtime = data_file.stat().st_mtime
    initial_line_count = count_lines(data_file)
    
    print(f"Initial state:")
    print(f"  File size: {initial_size} bytes")
    print(f"  Line count: {initial_line_count}")
    print(f"  Last modified: {datetime.fromtimestamp(initial_mtime)}")
    
    # Get last data row timestamp
    last_timestamp = get_last_timestamp(data_file)
    if last_timestamp:
        age_seconds = (datetime.now() - last_timestamp).total_seconds()
        print(f"  Last data timestamp: {last_timestamp} (age: {age_seconds:.1f}s)")
    else:
        print(f"  Last data timestamp: No valid data found")
    
    print("\nMonitoring for changes...")
    print("Time     | Size   | Lines | Last Price | Data Age | Status")
    print("-" * 65)
    
    start_time = time.time()
    changes_detected = 0
    last_check_size = initial_size
    last_check_lines = initial_line_count
    
    while time.time() - start_time < duration_seconds:
        try:
            # Check current state
            current_size = data_file.stat().st_size
            current_lines = count_lines(data_file)
            current_timestamp = get_last_timestamp(data_file)
            
            # Determine status
            if current_size > last_check_size:
                status = "GROWING"
                changes_detected += 1
            elif current_size == last_check_size:
                status = "STAGNANT"
            else:
                status = "SHRINKING"
            
            # Get current price and age
            current_price = "N/A"
            data_age = "N/A"
            
            if current_timestamp:
                data_age_seconds = (datetime.now() - current_timestamp).total_seconds()
                data_age = f"{data_age_seconds:.1f}s"
                
                # Try to get price from last line
                try:
                    last_line = get_last_data_line(data_file)
                    if last_line:
                        parts = last_line.split(',')
                        if len(parts) >= 4:
                            current_price = f"{float(parts[3]):.2f}"  # last price
                except:
                    pass
            
            # Print status line
            elapsed = time.time() - start_time
            print(f"{elapsed:6.1f}s | {current_size:6d} | {current_lines:5d} | {current_price:>10s} | {data_age:>8s} | {status}")
            
            # Update for next iteration
            last_check_size = current_size
            last_check_lines = current_lines
            
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            break
    
    # Final summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    final_size = data_file.stat().st_size if data_file.exists() else 0
    final_lines = count_lines(data_file) if data_file.exists() else 0
    
    size_growth = final_size - initial_size
    line_growth = final_lines - initial_line_count
    
    print(f"Initial size: {initial_size} bytes")
    print(f"Final size:   {final_size} bytes")
    print(f"Growth:       {size_growth} bytes ({line_growth} lines)")
    print(f"Changes detected: {changes_detected}")
    print(f"Test duration: {duration_seconds} seconds")
    
    # Determine test result
    if size_growth > 0 and line_growth > 0:
        print("✅ PASS: File is growing with new data")
        result = True
    elif changes_detected > 0:
        print("⚠️  PARTIAL: Some changes detected but file not growing consistently")
        result = False
    else:
        print("❌ FAIL: No data growth detected")
        result = False
    
    # Recommendations
    if not result:
        print("\nTROUBLESHOOTING:")
        print("1. Check if NinjaTrader is running and connected")
        print("2. Verify ESDataBridge indicator is applied to ES chart")
        print("3. Ensure ES chart is receiving live tick data")
        print("4. Check NinjaTrader Output window for error messages")
        print("5. Verify indicator is using correct file path")
    
    return result

def count_lines(file_path):
    """Count lines in file efficiently"""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def get_last_timestamp(file_path):
    """Get timestamp from last data line"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Look for last non-empty line that's not the header
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('timestamp'):
                parts = line.split(',')
                if len(parts) >= 1:
                    try:
                        return datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        # Try without microseconds
                        try:
                            return datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            continue
        
        return None
        
    except Exception as e:
        print(f"Error reading timestamp: {e}")
        return None

def get_last_data_line(file_path):
    """Get last data line from file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find last non-empty line that's not the header
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('timestamp'):
                return line
        
        return None
        
    except Exception as e:
        return None

def quick_file_check():
    """Quick check of current file state"""
    print("Quick File Check")
    print("=" * 20)
    
    data_file = Path(MARKET_DATA_FILE)
    
    if not data_file.exists():
        print("❌ File does not exist")
        return
    
    # File stats
    stat = data_file.stat()
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime)
    lines = count_lines(data_file)
    
    print(f"File: {data_file.name}")
    print(f"Size: {size} bytes")
    print(f"Lines: {lines}")
    print(f"Modified: {mtime}")
    
    # Last timestamp
    last_ts = get_last_timestamp(data_file)
    if last_ts:
        age = (datetime.now() - last_ts).total_seconds()
        print(f"Last data: {last_ts} (age: {age:.1f}s)")
        
        # Data freshness
        if age < 5:
            print("✅ Data is FRESH")
        elif age < 30:
            print("⚠️  Data is STALE")
        else:
            print("❌ Data is VERY STALE")
    else:
        print("❌ No valid data timestamps found")
    
    # Show last few lines
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\nLast 3 lines:")
        for line in lines[-3:]:
            print(f"  {line.rstrip()}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NinjaScript data flow')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds (default: 10)')
    parser.add_argument('--quick', action='store_true', help='Quick file check only')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_file_check()
    else:
        success = test_data_flow(args.duration)
        sys.exit(0 if success else 1)