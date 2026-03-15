#!/usr/bin/env python3
"""
Setup Script for NinjaTrader Bridge
Creates directory structure and tests file access
"""

import os
import logging
from pathlib import Path
from datetime import datetime

def setup_nt_bridge():
    """Setup NinjaTrader bridge directory and files"""
    
    print("NinjaTrader Bridge Setup")
    print("=" * 40)
    
    # Bridge directory
    bridge_dir = Path(r"C:\NTBridge")
    
    # Required files
    files_to_create = [
        bridge_dir / "market_data.csv",
        bridge_dir / "signals.txt", 
        bridge_dir / "status.txt",
        bridge_dir / "execution_log.txt"
    ]
    
    try:
        # Create main directory
        print(f"Creating bridge directory: {bridge_dir}")
        bridge_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Directory created successfully")
        
        # Test directory permissions
        test_file = bridge_dir / "permission_test.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("✓ Directory write permissions OK")
        except Exception as e:
            print(f"❌ Directory write permission error: {e}")
            return False
        
        # Create required files
        print("\\nCreating required files:")
        
        # Market data CSV with header
        market_data_file = bridge_dir / "market_data.csv"
        market_data_file.write_text("timestamp,bid,ask,last,volume,high,low,open,close\\n")
        print(f"✓ Created: {market_data_file}")
        
        # Empty signals file  
        signals_file = bridge_dir / "signals.txt"
        signals_file.touch()
        print(f"✓ Created: {signals_file}")
        
        # Status file
        status_file = bridge_dir / "status.txt"
        status_file.write_text(f"{datetime.now().isoformat()},SETUP,Bridge initialized\\n")
        print(f"✓ Created: {status_file}")
        
        # Execution log
        log_file = bridge_dir / "execution_log.txt"
        log_file.write_text(f"{datetime.now().isoformat()},INIT,Bridge setup completed\\n")
        print(f"✓ Created: {log_file}")
        
        # Test file operations
        print("\\nTesting file operations:")
        
        # Test market data write
        test_market_data = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},6595.00,6595.25,6595.12,100,6596.00,6594.50,6594.75,6595.12"
        market_data_file.write_text(test_market_data + "\\n")
        read_back = market_data_file.read_text().strip()
        if test_market_data in read_back:
            print("✓ Market data file read/write OK")
        else:
            print("❌ Market data file read/write failed")
            return False
        
        # Test signal write
        test_signal = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},BUY,1,0.4523"
        signals_file.write_text(test_signal)
        read_signal = signals_file.read_text().strip()
        if read_signal == test_signal:
            print("✓ Signal file read/write OK")
        else:
            print("❌ Signal file read/write failed")
            return False
        
        # Clear test data
        market_data_file.write_text("timestamp,bid,ask,last,volume,high,low,open,close\\n")
        signals_file.write_text("")
        
        print("\\n" + "=" * 40)
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("\\nNext steps:")
        print("1. Copy NinjaScript files to NinjaTrader:")
        print("   - ESDataBridge.cs → Documents/NinjaTrader 8/bin/Custom/Indicators/")
        print("   - ESSignalExecutor.cs → Documents/NinjaTrader 8/bin/Custom/Strategies/")
        print("2. Compile scripts in NinjaTrader NinjaScript Editor")  
        print("3. Add ESDataBridge indicator to your ES chart")
        print("4. Apply ESSignalExecutor strategy to your ES chart")
        print("5. Run integration test: python test_full_pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_nt_bridge()
    
    if success:
        print("\\n🎯 Bridge setup completed successfully!")
    else:
        print("\\n❌ Bridge setup failed!")
        print("Check permissions and try running as Administrator")
    
    input("\\nPress Enter to exit...")