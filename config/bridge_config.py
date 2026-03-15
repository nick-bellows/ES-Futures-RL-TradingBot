"""
Bridge Configuration
Centralized configuration for NinjaTrader bridge file paths
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path("D:/QC_TradingBot_v3")

# Bridge data directory within project
BRIDGE_DIR = PROJECT_ROOT / "data" / "bridge"

# Bridge file paths
MARKET_DATA_FILE = BRIDGE_DIR / "market_data.csv"
SIGNALS_FILE = BRIDGE_DIR / "signals.txt" 
STATUS_FILE = BRIDGE_DIR / "status.txt"
EXECUTION_LOG_FILE = BRIDGE_DIR / "execution_log.txt"

# NinjaScript paths (for documentation)
# Use these paths in your NinjaScript files:
NINJASCRIPT_PATHS = {
    'market_data': r"D:\QC_TradingBot_v3\data\bridge\market_data.csv",
    'signals': r"D:\QC_TradingBot_v3\data\bridge\signals.txt", 
    'status': r"D:\QC_TradingBot_v3\data\bridge\status.txt",
    'execution_log': r"D:\QC_TradingBot_v3\data\bridge\execution_log.txt"
}

# Bridge configuration
BRIDGE_CONFIG = {
    'max_data_age_seconds': 2.0,        # Emergency stop if data older than 2 seconds
    'signal_check_interval': 1.0,       # NinjaScript checks every 1 second
    'market_data_throttle_ms': 250,     # Write market data max every 250ms
    'signal_timeout_seconds': 10.0,     # Signals expire after 10 seconds
    'min_confidence_threshold': 0.3,    # Minimum signal confidence
    'bridge_directory': str(BRIDGE_DIR), # Bridge directory path
    'market_data_file': str(MARKET_DATA_FILE),
    'signals_file': str(SIGNALS_FILE),
    'status_file': str(STATUS_FILE),
    'execution_log_file': str(EXECUTION_LOG_FILE)
}

def ensure_bridge_directory():
    """Ensure bridge directory and structure exists"""
    try:
        # Create bridge directory
        BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create empty files if they don't exist
        for file_path in [MARKET_DATA_FILE, SIGNALS_FILE, STATUS_FILE, EXECUTION_LOG_FILE]:
            if not file_path.exists():
                file_path.touch()
        
        return True
        
    except Exception as e:
        print(f"Error creating bridge directory: {e}")
        return False

def get_bridge_info():
    """Get bridge configuration information"""
    return {
        'project_root': str(PROJECT_ROOT),
        'bridge_dir': str(BRIDGE_DIR),
        'files': {
            'market_data': str(MARKET_DATA_FILE),
            'signals': str(SIGNALS_FILE),
            'status': str(STATUS_FILE),
            'execution_log': str(EXECUTION_LOG_FILE)
        },
        'ninjascript_paths': NINJASCRIPT_PATHS,
        'config': BRIDGE_CONFIG,
        'bridge_exists': BRIDGE_DIR.exists(),
        'all_files_exist': all(f.exists() for f in [MARKET_DATA_FILE, SIGNALS_FILE, STATUS_FILE, EXECUTION_LOG_FILE])
    }

# Initialize bridge directory on import
ensure_bridge_directory()

if __name__ == "__main__":
    info = get_bridge_info()
    print("NinjaTrader Bridge Configuration")
    print("=" * 40)
    print(f"Project Root: {info['project_root']}")
    print(f"Bridge Directory: {info['bridge_dir']}")
    print(f"Bridge Exists: {info['bridge_exists']}")
    print(f"All Files Exist: {info['all_files_exist']}")
    print("\nBridge Files:")
    for name, path in info['files'].items():
        exists = Path(path).exists()
        print(f"  {name}: {path} {'✓' if exists else '❌'}")
    print("\nNinjaScript Paths (use these in C# code):")
    for name, path in info['ninjascript_paths'].items():
        print(f"  {name}: {path}")