#!/usr/bin/env python3
"""
Full Pipeline Integration Test
Tests the complete NinjaTrader file-based bridge system
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Import our bridge components
from integrations.ninjatrader_bridge.market_data_bridge import MarketDataBridge
from integrations.ninjatrader_bridge.signal_writer import SignalWriter

def test_full_pipeline():
    """Test the complete pipeline from market data to signal execution"""
    
    print("🧪 NinjaTrader Bridge Full Pipeline Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize components
    print("\\n1. Initializing components...")
    market_bridge = MarketDataBridge()
    signal_writer = SignalWriter()
    
    print("✓ Market data bridge initialized")
    print("✓ Signal writer initialized")
    
    # Test 1: Check bridge directory and files
    print("\\n2. Checking bridge setup...")
    bridge_dir = Path(r"C:\\NTBridge")
    
    required_files = {
        "market_data.csv": "Market data from ESDataBridge indicator",
        "signals.txt": "Trading signals for ESSignalExecutor strategy", 
        "status.txt": "Status updates from NinjaScript",
        "execution_log.txt": "Execution log from strategy"
    }
    
    setup_ok = True
    for filename, description in required_files.items():
        filepath = bridge_dir / filename
        if filepath.exists():
            print(f"✓ {filename} exists - {description}")
        else:
            print(f"❌ {filename} missing - {description}")
            setup_ok = False
    
    if not setup_ok:
        print("\\n❌ Bridge setup incomplete. Run: python setup_nt_bridge.py")
        return False
    
    # Test 2: Market data health check
    print("\\n3. Market data health check...")
    health_check = market_bridge.emergency_check()
    
    print(f"Status: {health_check['status']}")
    print(f"Current Price: {health_check['current_price']:.2f}")
    print(f"Data Age: {health_check['data_age']:.1f} seconds")
    print(f"File Exists: {health_check['file_exists']}")
    
    if health_check['issues']:
        print("Issues found:")
        for issue in health_check['issues']:
            print(f"  • {issue}")
    
    if health_check['recommendations']:
        print("Recommendations:")
        for rec in health_check['recommendations']:
            print(f"  • {rec}")
    
    if health_check['status'] == 'CRITICAL':
        print("\\n❌ Market data is not available!")
        print("Make sure:")
        print("  1. NinjaTrader is running")
        print("  2. ESDataBridge indicator is added to your ES chart")
        print("  3. Chart is connected to real-time data")
        return False
    
    # Test 3: Start monitoring and test data flow
    if health_check['status'] in ['OK', 'WARNING']:
        print("\\n4. Testing live data flow...")
        
        if market_bridge.start_monitoring():
            print("✓ Market data monitoring started")
            
            # Monitor for 10 seconds
            start_time = time.time()
            tick_count = 0
            last_price = 0
            
            while time.time() - start_time < 10:
                tick = market_bridge.get_latest_tick()
                if tick:
                    current_price = tick.last
                    if current_price != last_price:
                        tick_count += 1
                        age = market_bridge.get_data_age_seconds()
                        print(f"  Tick {tick_count}: {current_price:.2f} (age: {age:.1f}s)")
                        last_price = current_price
                
                time.sleep(0.5)
            
            market_bridge.stop_monitoring()
            print(f"✓ Received {tick_count} price updates in 10 seconds")
            
            if tick_count == 0:
                print("⚠️  No price updates received - check if market is open")
            elif tick_count < 5:
                print("⚠️  Low update frequency - acceptable for testing")
            else:
                print("✓ Good update frequency")
        else:
            print("❌ Failed to start market data monitoring")
            return False
    
    # Test 4: Signal writing and round-trip latency
    print("\\n5. Testing signal writing...")
    
    test_signals = [
        ("BUY", 6595.50, 0.4523),
        ("SELL", 6594.25, 0.3891), 
        ("FLAT", 6595.00, 1.0)
    ]
    
    latency_times = []
    
    for action, price, confidence in test_signals:
        print(f"\\nTesting {action} signal at {price:.2f}...")
        
        # Measure write latency
        start_time = time.time()
        
        if action == "BUY":
            success = signal_writer.write_buy_signal(price, confidence)
        elif action == "SELL":
            success = signal_writer.write_sell_signal(price, confidence)
        else:  # FLAT
            success = signal_writer.write_flat_signal(price, confidence)
        
        write_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        latency_times.append(write_time)
        
        if success:
            print(f"✓ {action} signal written in {write_time:.1f}ms")
            
            # Verify signal file content
            last_signal = signal_writer.get_last_signal()
            if last_signal and action in last_signal:
                print(f"✓ Signal verified in file: {last_signal}")
            else:
                print(f"⚠️  Signal file content unexpected: {last_signal}")
        else:
            print(f"❌ Failed to write {action} signal")
        
        time.sleep(2)  # Wait between signals
    
    # Calculate average latency
    if latency_times:
        avg_latency = sum(latency_times) / len(latency_times)
        max_latency = max(latency_times)
        print(f"\\n📊 Signal Writing Performance:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Maximum latency: {max_latency:.1f}ms")
        
        if avg_latency < 10:
            print("✓ Excellent latency (<10ms)")
        elif avg_latency < 50:
            print("✓ Good latency (<50ms)")
        elif avg_latency < 100:
            print("⚠️  Acceptable latency (<100ms)")
        else:
            print("❌ High latency (>100ms) - may impact trading performance")
    
    # Test 5: Check NinjaScript execution
    print("\\n6. Checking NinjaScript execution...")
    
    execution_log_file = Path(r"C:\\NTBridge\\execution_log.txt")
    if execution_log_file.exists():
        # Read recent log entries
        try:
            log_content = execution_log_file.read_text()
            log_lines = log_content.strip().split('\\n')
            recent_logs = log_lines[-5:] if len(log_lines) >= 5 else log_lines
            
            print("Recent execution log entries:")
            for log_line in recent_logs:
                if log_line.strip():
                    print(f"  {log_line}")
            
            # Check for recent activity
            if log_lines:
                last_log = log_lines[-1]
                if ',' in last_log:
                    timestamp_str = last_log.split(',')[0]
                    try:
                        last_activity = datetime.fromisoformat(timestamp_str)
                        age = (datetime.now() - last_activity).total_seconds()
                        
                        if age < 60:
                            print(f"✓ Recent NinjaScript activity ({age:.0f}s ago)")
                        else:
                            print(f"⚠️  Last NinjaScript activity was {age:.0f}s ago")
                    except:
                        print("⚠️  Could not parse log timestamp")
            else:
                print("⚠️  No execution log entries found")
                
        except Exception as e:
            print(f"❌ Error reading execution log: {e}")
    else:
        print("⚠️  Execution log file not found")
        print("   Make sure ESSignalExecutor strategy is running")
    
    # Test 6: Overall system health
    print("\\n7. Overall system health check...")
    
    # Get statistics
    market_stats = market_bridge.get_statistics()
    signal_stats = signal_writer.get_signal_stats()
    
    print("Market Data Bridge:")
    print(f"  Ticks processed: {market_stats['ticks_processed']}")
    print(f"  Read errors: {market_stats['read_errors']}")
    print(f"  Current price: {market_stats['current_price']:.2f}")
    print(f"  Data is fresh: {market_stats['data_is_fresh']}")
    
    print("Signal Writer:")
    print(f"  Signals written: {signal_stats['signal_count']}")
    print(f"  Last signal time: {signal_stats['last_signal_time']}")
    
    # Final assessment
    print("\\n" + "=" * 50)
    
    issues_found = []
    warnings_found = []
    
    if health_check['status'] == 'CRITICAL':
        issues_found.append("Market data not available")
    elif health_check['status'] == 'WARNING':
        warnings_found.append("Market data has minor issues")
    
    if not market_stats['data_is_fresh']:
        issues_found.append("Market data is stale")
    
    if signal_stats['signal_count'] == 0:
        issues_found.append("No signals were written")
    
    if avg_latency > 100:
        warnings_found.append("High signal writing latency")
    
    # Final result
    if issues_found:
        print("❌ PIPELINE TEST FAILED")
        print("Critical issues found:")
        for issue in issues_found:
            print(f"  • {issue}")
    elif warnings_found:
        print("⚠️  PIPELINE TEST PASSED WITH WARNINGS")
        print("Warnings:")
        for warning in warnings_found:
            print(f"  • {warning}")
    else:
        print("✅ PIPELINE TEST PASSED!")
        print("All systems are working correctly")
    
    if issues_found or warnings_found:
        print("\\nTroubleshooting steps:")
        print("1. Ensure NinjaTrader is running and connected")
        print("2. Check that ESDataBridge indicator is on your chart")
        print("3. Verify ESSignalExecutor strategy is applied")
        print("4. Confirm market hours (ES trades nearly 24/7)")
        print("5. Check Windows file permissions for C:\\NTBridge\\")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        
        if success:
            print("\\n🎯 Integration test completed successfully!")
            print("Your NinjaTrader bridge is ready for live trading.")
        else:
            print("\\n❌ Integration test failed!")
            print("Fix the issues above before using the trading bot.")
            
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\\nPress Enter to exit...")