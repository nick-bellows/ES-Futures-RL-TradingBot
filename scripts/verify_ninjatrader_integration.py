#!/usr/bin/env python3
"""
NinjaTrader Integration Verification Script
Helps verify that ESSignalExecutor.cs is properly installed and functioning
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def check_signal_file():
    """Check if signal file exists and has recent content"""
    signal_file = Path("D:/QC_TradingBot_v3/data/bridge/signals.txt")
    
    print("=" * 60)
    print("1. CHECKING SIGNAL FILE")
    print("=" * 60)
    
    if not signal_file.exists():
        print("[ERROR] Signal file does not exist:")
        print(f"        {signal_file}")
        print("        Run the Python trading bot first to create signals.")
        return False
    
    print(f"[OK] Signal file exists: {signal_file}")
    
    # Check file age
    stat = signal_file.stat()
    file_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
    print(f"[INFO] File last modified: {datetime.fromtimestamp(stat.st_mtime)}")
    print(f"[INFO] File age: {file_age.total_seconds():.1f} seconds")
    
    # Check file content
    try:
        with open(signal_file, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print("[WARNING] Signal file is empty")
            print("          Run the Python trading bot to generate signals")
            return False
            
        print(f"[OK] Signal content: {content}")
        
        # Parse signal format
        parts = content.split(',')
        if len(parts) != 4:
            print(f"[ERROR] Invalid signal format - expected 4 parts, got {len(parts)}")
            return False
        
        print(f"[OK] Signal format valid:")
        print(f"      Timestamp: {parts[0]}")
        print(f"      Action: {parts[1]}")
        print(f"      Quantity: {parts[2]}")
        print(f"      Confidence: {parts[3]}")
        
        # Check signal age
        try:
            signal_time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S.%f")
            signal_age = datetime.now() - signal_time
            print(f"[INFO] Signal age: {signal_age.total_seconds():.1f} seconds")
            
            if signal_age.total_seconds() > 300:  # 5 minutes
                print("[WARNING] Signal is quite old - may need fresh signals")
            
        except ValueError as e:
            print(f"[ERROR] Could not parse signal timestamp: {e}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not read signal file: {e}")
        return False

def check_ninjatrader_files():
    """Check for NinjaTrader installation and ESSignalExecutor"""
    
    print("\n" + "=" * 60)
    print("2. CHECKING NINJATRADER INSTALLATION")
    print("=" * 60)
    
    # Common NinjaTrader paths
    nt_paths = [
        Path.home() / "Documents" / "NinjaTrader 8",
        Path("C:/Program Files/NinjaTrader 8"),
        Path("C:/Program Files (x86)/NinjaTrader 8"),
    ]
    
    nt_path = None
    for path in nt_paths:
        if path.exists():
            nt_path = path
            print(f"[OK] Found NinjaTrader 8 at: {path}")
            break
    
    if not nt_path:
        print("[ERROR] NinjaTrader 8 installation not found!")
        print("        Checked these locations:")
        for path in nt_paths:
            print(f"        - {path}")
        return False
    
    # Check for custom strategies folder
    strategies_path = nt_path / "bin" / "Custom" / "Strategies"
    if not strategies_path.exists():
        print(f"[ERROR] Custom strategies folder not found: {strategies_path}")
        return False
    
    print(f"[OK] Custom strategies folder: {strategies_path}")
    
    # Check for ESSignalExecutor.cs
    executor_path = strategies_path / "ESSignalExecutor.cs"
    if not executor_path.exists():
        print(f"[ERROR] ESSignalExecutor.cs not found at: {executor_path}")
        print("        You need to copy the NinjaScript file to this location.")
        print("        Source: D:/QC_TradingBot_v3/ninjascript/UpdatedESSignalExecutor.cs")
        return False
    
    print(f"[OK] ESSignalExecutor.cs found: {executor_path}")
    
    # Check file age to see if it's recent
    stat = executor_path.stat()
    file_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
    print(f"[INFO] ESSignalExecutor last modified: {datetime.fromtimestamp(stat.st_mtime)}")
    print(f"[INFO] File age: {file_age.total_seconds():.0f} seconds")
    
    return True

def check_execution_log():
    """Check for NinjaTrader execution log"""
    
    print("\n" + "=" * 60)  
    print("3. CHECKING EXECUTION LOG")
    print("=" * 60)
    
    log_file = Path("D:/QC_TradingBot_v3/data/bridge/execution_log.txt")
    
    if not log_file.exists():
        print("[WARNING] Execution log file does not exist:")
        print(f"          {log_file}")
        print("          This suggests ESSignalExecutor has not run yet.")
        return False
    
    print(f"[OK] Execution log exists: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("[WARNING] Execution log is empty")
            return False
        
        print(f"[OK] Log has {len(lines)} entries")
        
        # Show recent entries
        recent_lines = lines[-5:]  # Last 5 entries
        print("[INFO] Recent log entries:")
        for line in recent_lines:
            print(f"       {line.strip()}")
        
        # Check for recent activity
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                try:
                    timestamp_str = last_line.split(',')[0]
                    last_activity = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    age = datetime.now() - last_activity
                    print(f"[INFO] Last activity: {age.total_seconds():.1f} seconds ago")
                    
                    if age.total_seconds() > 300:  # 5 minutes
                        print("[WARNING] No recent activity in execution log")
                        
                except Exception as e:
                    print(f"[WARNING] Could not parse last log entry timestamp: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not read execution log: {e}")
        return False

def provide_troubleshooting_guide():
    """Provide troubleshooting steps"""
    
    print("\n" + "=" * 60)
    print("4. TROUBLESHOOTING GUIDE")
    print("=" * 60)
    
    print("If NinjaTrader is not processing signals, check these steps:")
    print()
    
    print("STEP 1: Verify ESSignalExecutor Installation")
    print("- Copy D:/QC_TradingBot_v3/ninjascript/UpdatedESSignalExecutor.cs")
    print("- To: Documents/NinjaTrader 8/bin/Custom/Strategies/ESSignalExecutor.cs")
    print("- Restart NinjaTrader to compile the strategy")
    print()
    
    print("STEP 2: Apply Strategy to ES Chart")
    print("- Open NinjaTrader")  
    print("- Create/Open ES 12-25 (ESZ25) chart")
    print("- Right-click chart -> Strategies -> ESSignalExecutor")
    print("- Apply with default settings")
    print()
    
    print("STEP 3: Enable Strategy")
    print("- Make sure strategy is ENABLED (not just applied)")
    print("- Check Chart Trader is connected to live account")
    print("- Verify ES 12-25 contract is available")
    print()
    
    print("STEP 4: Check NinjaTrader Output Window")
    print("- Open NinjaTrader Output window (Ctrl+O)")
    print("- Look for '[DEBUG]' messages from ESSignalExecutor")
    print("- Should see messages like '[DEBUG] Checking signals' every 10 seconds")
    print("- Should see '[DEBUG] Read signal data' when processing signals")
    print()
    
    print("STEP 5: Check Trading Permissions")
    print("- Verify account has ES futures trading permission")
    print("- Check sufficient buying power")
    print("- Ensure market is open (ES trades nearly 24/5)")
    print()
    
    print("STEP 6: Monitor Debug Output")
    print("- Watch NinjaTrader Output window while Python bot runs")
    print("- Debug messages will show exact reason signals are rejected")
    print("- Common issues: stale signals, low confidence, position limits")

def run_live_test():
    """Run a live test by generating a fresh signal"""
    
    print("\n" + "=" * 60)
    print("5. LIVE INTEGRATION TEST")
    print("=" * 60)
    
    print("Testing signal generation and NinjaTrader response...")
    
    try:
        # Import signal writer
        import sys
        sys.path.append(str(Path(__file__).parent))
        from ninjatrader_bridge.signal_writer import ProjectSignalWriter
        
        writer = ProjectSignalWriter()
        
        # Generate test signal
        test_price = 5500.0  # Safe test price for ES
        test_confidence = 0.45  # Above 35% threshold
        
        print(f"[TEST] Writing BUY signal @ {test_price:.2f} (conf: {test_confidence:.1%})")
        success = writer.write_buy_signal(test_price, test_confidence, 1)
        
        if success:
            print("[OK] Test signal written successfully")
            print("     Check NinjaTrader Output window for processing messages")
            print("     You should see debug output within 1-10 seconds")
        else:
            print("[ERROR] Failed to write test signal")
            stats = writer.get_signal_stats()
            if stats['last_error']:
                print(f"        Error: {stats['last_error']}")
        
        # Show current signal content
        last_signal = writer.get_last_signal()
        if last_signal:
            print(f"[INFO] Current signal in file: {last_signal}")
            
        return success
        
    except Exception as e:
        print(f"[ERROR] Live test failed: {e}")
        return False

def main():
    """Run complete verification"""
    
    print("NinjaTrader Integration Verification")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    results = []
    
    # Run all checks
    results.append(("Signal File", check_signal_file()))
    results.append(("NinjaTrader Installation", check_ninjatrader_files()))
    results.append(("Execution Log", check_execution_log()))
    results.append(("Live Test", run_live_test()))
    
    # Show troubleshooting guide
    provide_troubleshooting_guide()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nOverall: {total_passed}/{len(results)} checks passed")
    
    if total_passed == len(results):
        print("\n[SUCCESS] All checks passed!")
        print("If NinjaTrader still isn't trading, check the Output window for debug messages.")
    else:
        print("\n[ATTENTION] Some checks failed.")
        print("Follow the troubleshooting guide above to resolve issues.")
    
    return total_passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)