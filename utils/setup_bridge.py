"""
NinjaTrader Bridge Setup (Clean Version - No Unicode)
Creates and initializes project-based bridge directory structure
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.bridge_config import (
    BRIDGE_DIR, 
    MARKET_DATA_FILE, 
    SIGNALS_FILE, 
    STATUS_FILE, 
    EXECUTION_LOG_FILE,
    BRIDGE_CONFIG
)

def setup_project_bridge():
    """Setup complete project-based bridge system"""
    try:
        print("NinjaTrader Project Bridge Setup")
        print("=" * 50)
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Bridge Directory: {BRIDGE_DIR}")
        print()
        
        # Step 1: Create bridge directory
        print("Creating bridge directory structure...")
        BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"CREATED: {BRIDGE_DIR}")
        
        # Step 2: Initialize bridge files
        print("\nInitializing bridge files...")
        
        # Market data file with CSV header
        if not MARKET_DATA_FILE.exists():
            with open(MARKET_DATA_FILE, 'w') as f:
                f.write("timestamp,bid,ask,last,volume,high,low,open,close\n")
            print(f"CREATED: {MARKET_DATA_FILE.name}")
        else:
            print(f"EXISTS: {MARKET_DATA_FILE.name}")
        
        # Signals file (empty)
        if not SIGNALS_FILE.exists():
            SIGNALS_FILE.touch()
            print(f"CREATED: {SIGNALS_FILE.name}")
        else:
            print(f"EXISTS: {SIGNALS_FILE.name}")
        
        # Status file with initial status
        with open(STATUS_FILE, 'w') as f:
            f.write(f"{datetime.now().isoformat()},INITIALIZED,Bridge setup completed\n")
        print(f"INITIALIZED: {STATUS_FILE.name}")
        
        # Execution log file with header
        if not EXECUTION_LOG_FILE.exists():
            with open(EXECUTION_LOG_FILE, 'w') as f:
                f.write(f"{datetime.now().isoformat()},SETUP,Bridge initialized\n")
            print(f"CREATED: {EXECUTION_LOG_FILE.name}")
        else:
            # Append setup message
            with open(EXECUTION_LOG_FILE, 'a') as f:
                f.write(f"{datetime.now().isoformat()},SETUP,Bridge reinitialized\n")
            print(f"UPDATED: {EXECUTION_LOG_FILE.name}")
        
        # Step 3: Test file permissions
        print("\nTesting file permissions...")
        
        test_files = [MARKET_DATA_FILE, SIGNALS_FILE, STATUS_FILE, EXECUTION_LOG_FILE]
        for test_file in test_files:
            try:
                # Test read
                with open(test_file, 'r') as f:
                    f.read(100)  # Read first 100 chars
                
                # Test write
                with open(test_file, 'a') as f:
                    f.write("")  # Write empty string
                
                print(f"READ/WRITE OK: {test_file.name}")
                
            except Exception as e:
                print(f"PERMISSION ERROR: {test_file.name} - {e}")
                return False
        
        # Step 4: Create test data
        print("\nCreating test data...")
        
        # Test market data
        test_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        test_market_data = f"{test_timestamp},6595.50,6595.75,6595.62,1234,6600.00,6590.00,6595.00,6595.62"
        
        with open(MARKET_DATA_FILE, 'w') as f:
            f.write("timestamp,bid,ask,last,volume,high,low,open,close\n")
            f.write(test_market_data + "\n")
        
        print("TEST MARKET DATA: Created")
        
        # Test signal
        test_signal = f"{test_timestamp},BUY,1,0.4523"
        with open(SIGNALS_FILE, 'w') as f:
            f.write(test_signal)
        
        print("TEST SIGNAL: Created")
        
        # Step 5: Verification
        print("\nVerifying setup...")
        
        all_files_exist = all(f.exists() for f in test_files)
        all_files_writable = True
        
        for test_file in test_files:
            if not os.access(test_file, os.W_OK):
                all_files_writable = False
                print(f"NOT WRITABLE: {test_file}")
        
        if all_files_exist and all_files_writable:
            print("VERIFICATION: PASSED")
        else:
            print("VERIFICATION: FAILED")
            return False
        
        # Step 6: Display setup summary
        print("\n" + "=" * 50)
        print("BRIDGE SETUP COMPLETE")
        print("=" * 50)
        print(f"Bridge Directory: {BRIDGE_DIR}")
        print("Files Created:")
        print(f"  Market Data: {MARKET_DATA_FILE}")
        print(f"  Signals:     {SIGNALS_FILE}")
        print(f"  Status:      {STATUS_FILE}")
        print(f"  Exec Log:    {EXECUTION_LOG_FILE}")
        print()
        print("NEXT STEPS:")
        print("1. Copy NinjaScript files to NinjaTrader:")
        print("   - UpdatedESDataBridge.cs to Custom\\Indicators\\")
        print("   - UpdatedESSignalExecutor.cs to Custom\\Strategies\\")
        print("2. Compile NinjaScript files (F5)")
        print("3. Add ESDataBridge indicator to ES chart")
        print("4. Add ESSignalExecutor strategy to same chart")
        print("5. Run integration test: python tests/test_bridge_integration.py")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

def clean_bridge():
    """Clean/reset bridge files"""
    try:
        print("Cleaning bridge files...")
        
        files_to_clean = [MARKET_DATA_FILE, SIGNALS_FILE]
        
        for file_path in files_to_clean:
            if file_path.exists():
                file_path.unlink()
                print(f"REMOVED: {file_path.name}")
        
        print("Bridge files cleaned")
        return True
        
    except Exception as e:
        print(f"Clean failed: {e}")
        return False

def check_bridge_status():
    """Check current bridge status"""
    print("Bridge Status Check")
    print("=" * 30)
    print(f"Bridge Directory: {BRIDGE_DIR}")
    print(f"Directory Exists: {BRIDGE_DIR.exists()}")
    print()
    
    files = {
        'Market Data': MARKET_DATA_FILE,
        'Signals': SIGNALS_FILE,
        'Status': STATUS_FILE,
        'Execution Log': EXECUTION_LOG_FILE
    }
    
    for name, file_path in files.items():
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        print(f"{name:15}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
    
    print()
    print("Configuration:")
    for key, value in BRIDGE_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NinjaTrader Bridge Setup')
    parser.add_argument('--clean', action='store_true', help='Clean bridge files')
    parser.add_argument('--status', action='store_true', help='Check bridge status')
    
    args = parser.parse_args()
    
    if args.clean:
        success = clean_bridge()
    elif args.status:
        check_bridge_status()
        success = True
    else:
        success = setup_project_bridge()
    
    sys.exit(0 if success else 1)