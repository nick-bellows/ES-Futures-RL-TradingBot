"""
NinjaTrader Configuration
Connection settings and trading parameters for NT8 integration
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class NTConnectionSettings:
    """NinjaTrader API connection configuration"""
    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 2.0
    
    # API endpoints
    base_url: str = "http://localhost:8080"
    market_data_endpoint: str = "/NTDataService/MarketData"
    order_endpoint: str = "/NTDataService/Order"
    position_endpoint: str = "/NTDataService/Position"
    account_endpoint: str = "/NTDataService/Account"
    
    def get_endpoint_url(self, endpoint_type: str) -> str:
        """Get full URL for specific endpoint"""
        endpoints = {
            'market_data': self.market_data_endpoint,
            'order': self.order_endpoint,
            'position': self.position_endpoint,
            'account': self.account_endpoint
        }
        
        if endpoint_type not in endpoints:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
            
        return f"{self.base_url}{endpoints[endpoint_type]}"

@dataclass
class NTTradingParameters:
    """ES Futures trading strategy parameters"""
    # Contract specifications
    instrument: str = "ES 09-25"  # Current contract
    tick_size: float = 0.25
    tick_value: float = 12.50  # $12.50 per tick
    point_value: float = 50.0  # $50 per point
    
    # Risk management (same as validated model)
    stop_loss_points: float = 5.0   # 20 ticks
    take_profit_points: float = 15.0  # 60 ticks
    confidence_threshold: float = 0.35
    max_daily_trades: int = 5
    max_daily_loss: float = 750.0
    
    # Position sizing
    contracts_per_trade: int = 1
    max_position_size: int = 1
    
    # Execution costs
    commission_per_rt: float = 2.50
    slippage_ticks: int = 1
    
    # Model parameters
    lookback_bars: int = 60
    feature_count: int = 47
    
    def get_stop_loss_ticks(self) -> int:
        """Convert stop loss points to ticks"""
        return int(self.stop_loss_points / self.tick_size)
    
    def get_take_profit_ticks(self) -> int:
        """Convert take profit points to ticks"""
        return int(self.take_profit_points / self.tick_size)

@dataclass
class NTAccountSettings:
    """NinjaTrader account configuration"""
    account_name: str = "YOUR_ACCOUNT_NAME"  # Set to your NinjaTrader account name
    account_type: str = "SIMULATION"
    currency: str = "USD"
    
    # Data feed settings
    data_provider: str = "NinjaTrader"
    bar_type: str = "Minute"
    bar_period: int = 1
    
    # Market hours (ET)
    market_open: str = "18:00"  # Sunday 6:00 PM ET
    market_close: str = "17:00"  # Friday 5:00 PM ET
    regular_open: str = "09:30"  # 9:30 AM ET  
    regular_close: str = "16:00"  # 4:00 PM ET

class NTConfig:
    """Main configuration class for NinjaTrader integration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "ninjatrader_config.json"
        
        # Initialize settings
        self.connection = NTConnectionSettings()
        self.trading = NTTradingParameters()
        self.account = NTAccountSettings()
        
        # System settings
        self.log_level: str = "INFO"
        self.log_file: str = "ninjatrader_integration.log"
        
        # Model paths (use existing validated paths)
        self.model_path: str = "models/best/ppo/ppo_best/best_model"
        self.scaler_path: str = "models/feature_scaler.pkl"
        
        # Results and logging
        self.results_directory: str = "ninjatrader_results"
        self.trade_log_file: str = "ninjatrader_trades.csv"
        
        # Contract rollover settings
        self.auto_rollover: bool = True
        self.rollover_days_ahead: int = 7  # Switch 7 days before expiry
        
        # Load existing config if available
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_file):
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update connection settings
                if 'connection' in config_data:
                    conn_data = config_data['connection']
                    for key, value in conn_data.items():
                        if hasattr(self.connection, key):
                            setattr(self.connection, key, value)
                
                # Update trading parameters
                if 'trading' in config_data:
                    trading_data = config_data['trading']
                    for key, value in trading_data.items():
                        if hasattr(self.trading, key):
                            setattr(self.trading, key, value)
                
                # Update account settings
                if 'account' in config_data:
                    account_data = config_data['account']
                    for key, value in account_data.items():
                        if hasattr(self.account, key):
                            setattr(self.account, key, value)
                
                # Update system settings
                self.log_level = config_data.get('log_level', self.log_level)
                self.model_path = config_data.get('model_path', self.model_path)
                self.auto_rollover = config_data.get('auto_rollover', self.auto_rollover)
                
                print(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                print(f"Failed to load config: {e}, using defaults")
        else:
            print("No config file found, using defaults")
            self.save_config()
    
    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        try:
            import json
            from dataclasses import asdict
            
            config_data = {
                'connection': asdict(self.connection),
                'trading': asdict(self.trading),
                'account': asdict(self.account),
                'log_level': self.log_level,
                'model_path': self.model_path,
                'scaler_path': self.scaler_path,
                'results_directory': self.results_directory,
                'auto_rollover': self.auto_rollover,
                'rollover_days_ahead': self.rollover_days_ahead
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def get_current_contract(self) -> str:
        """Get current ES contract based on date"""
        from datetime import datetime, date
        
        # ES contract months: March (H), June (M), September (U), December (Z)
        current_date = date.today()
        year = current_date.year % 100  # Get 2-digit year
        
        if current_date.month <= 3:
            return f"ES 03-{year:02d}"  # March
        elif current_date.month <= 6:
            return f"ES 06-{year:02d}"  # June
        elif current_date.month <= 9:
            return f"ES 09-{year:02d}"  # September
        else:
            next_year = (year + 1) % 100
            return f"ES 12-{next_year:02d}"  # December
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        
        return {
            'connection': asdict(self.connection),
            'trading': asdict(self.trading),
            'account': asdict(self.account),
            'system': {
                'log_level': self.log_level,
                'model_path': self.model_path,
                'scaler_path': self.scaler_path,
                'auto_rollover': self.auto_rollover
            }
        }