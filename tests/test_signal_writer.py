#!/usr/bin/env python3
"""
Test Signal Writer
Quick test to verify signal writing functionality is working properly
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from ninjatrader_bridge.signal_writer import ProjectSignalWriter


def test_signal_writer():
    """Test signal writer functionality"""
    print("Testing Signal Writer")
    print("=" * 50)
    
    # Create signal writer
    writer = ProjectSignalWriter()
    
    # Get initial stats
    stats = writer.get_signal_stats()
    print(f"Signal file path: {stats['signal_file_path']}")
    print(f"Bridge directory exists: {stats['bridge_directory_exists']}")
    print(f"Signal file exists: {stats['signal_file_exists']}")
    print(f"Signal file writable: {stats['signal_file_writable']}")
    print(f"Signal file size: {stats['signal_file_size']} bytes")
    print()
    
    if not stats['signal_file_writable']:
        print("[ERROR] Signal file is not writable!")
        return False
    
    # Test signals
    test_price = 6672.13
    test_signals = [
        ("BUY", 0.45),
        ("SELL", 0.42),
        ("BUY", 0.38),
    ]
    
    print("Testing signal writes...")
    for i, (action, confidence) in enumerate(test_signals):
        print(f"\\nTest {i+1}: {action} @ {test_price:.2f} (conf: {confidence:.1%})")
        
        try:
            success = writer.write_signal(action, test_price, confidence)
            if success:
                print(f"  [OK] SUCCESS: Signal written")
            else:
                print(f"  [ERROR] FAILED: Signal not written")
                stats = writer.get_signal_stats()
                if stats['last_error']:
                    print(f"    Error: {stats['last_error']}")
        except Exception as e:
            print(f"  [ERROR] EXCEPTION: {e}")
        
        # Check if signal was actually written
        last_signal = writer.get_last_signal()
        if last_signal:
            print(f"  File content: {last_signal}")
        else:
            print(f"  File is empty or unreadable")
        
        # Small delay between signals
        time.sleep(1.1)
    
    # Final stats
    print(f"\\nFinal Statistics:")
    final_stats = writer.get_signal_stats()
    print(f"  Signals written: {final_stats['signal_count']}")
    print(f"  Write failures: {final_stats['write_failures']}")
    if final_stats['last_error']:
        print(f"  Last error: {final_stats['last_error']}")
    
    return final_stats['signal_count'] > 0


def test_direct_file_write():
    """Test direct file writing to diagnose permission issues"""
    print("\\nTesting direct file operations...")
    print("=" * 50)
    
    signal_path = Path("D:/QC_TradingBot_v3/data/bridge/signals.txt")
    
    try:
        # Test directory creation
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        print("[OK] Directory creation successful")
    except Exception as e:
        print(f"[ERROR] Directory creation failed: {e}")
        return False
    
    try:
        # Test file creation/writing
        with open(signal_path, 'w') as f:
            f.write("TEST SIGNAL")
        print("[OK] File write successful")
    except Exception as e:
        print(f"[ERROR] File write failed: {e}")
        return False
    
    try:
        # Test file reading
        with open(signal_path, 'r') as f:
            content = f.read()
        print(f"[OK] File read successful: '{content}'")
    except Exception as e:
        print(f"[ERROR] File read failed: {e}")
        return False
    
    try:
        # Test file deletion
        signal_path.unlink()
        print("[OK] File deletion successful")
    except Exception as e:
        print(f"[ERROR] File deletion failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("Signal Writer Diagnostic Test")
    print("=" * 60)
    
    # Test 1: Direct file operations
    if not test_direct_file_write():
        print("\\n[CRITICAL] Basic file operations failed!")
        print("Check file permissions and disk space.")
        return False
    
    # Test 2: Signal writer
    if test_signal_writer():
        print("\\n[SUCCESS] Signal writer is working correctly!")
        return True
    else:
        print("\\n[ERROR] Signal writer has issues!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)