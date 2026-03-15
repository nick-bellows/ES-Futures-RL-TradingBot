#!/usr/bin/env python3
"""
Signal Monitor
Monitors signal file and shows what signals are being written in real-time
"""

import time
import os
from pathlib import Path


def monitor_signals(duration=60):
    """Monitor signals file for changes"""
    signal_file = Path("D:/QC_TradingBot_v3/data/bridge/signals.txt")
    
    print(f"Monitoring signals file: {signal_file}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    last_content = ""
    last_modified = 0
    signal_count = 0
    
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Check duration
            if duration and (current_time - start_time) > duration:
                print(f"\\nMonitoring complete after {duration} seconds")
                break
            
            try:
                if signal_file.exists():
                    # Check if file was modified
                    stat = signal_file.stat()
                    if stat.st_mtime != last_modified:
                        last_modified = stat.st_mtime
                        
                        # Read content
                        with open(signal_file, 'r') as f:
                            content = f.read().strip()
                        
                        if content != last_content:
                            signal_count += 1
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] Signal #{signal_count}: {content}")
                            last_content = content
                else:
                    if last_content != "":
                        print(f"[{time.strftime('%H:%M:%S')}] Signal file deleted")
                        last_content = ""
                        
            except Exception as e:
                print(f"Error reading signal file: {e}")
            
            time.sleep(0.1)  # Check every 100ms
            
    except KeyboardInterrupt:
        print(f"\\nMonitoring stopped. Total signals seen: {signal_count}")


if __name__ == "__main__":
    import sys
    
    duration = 60  # Default 60 seconds
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            print("Usage: python monitor_signals.py [duration_seconds]")
            sys.exit(1)
    
    monitor_signals(duration)