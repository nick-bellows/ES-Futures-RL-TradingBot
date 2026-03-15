"""
ES Futures Contract Rolling Utility
Detects roll periods and provides contract switching recommendations
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.contract_config import (
    ESContract, determine_current_contract, get_next_contract,
    get_contract_status, ES_CONTRACTS_2024_2026, display_contract_info
)

class ContractRoller:
    """
    ES Futures Contract Rolling Manager
    
    Handles detection of roll periods and provides switching recommendations
    """
    
    def __init__(self):
        self.current_contract = determine_current_contract()
        self.roll_history_file = PROJECT_ROOT / "data" / "contract_roll_history.json"
        self.roll_history = self._load_roll_history()
    
    def check_roll_status(self) -> Dict[str, any]:
        """
        Check current roll status and recommendations
        
        Returns:
            Dictionary with roll status and recommendations
        """
        current = self.current_contract
        status = get_contract_status(current)
        next_contract = status['next_contract']
        
        roll_info = {
            'current_contract': current.symbol,
            'current_code': current.code,
            'days_until_expiry': status['days_until_expiry'],
            'status': status['status'],
            'urgency': status['urgency'],
            'should_roll': False,
            'must_roll': False,
            'recommendation': '',
            'next_contract': None,
            'roll_date_suggestion': None,
            'warnings': []
        }
        
        if next_contract:
            roll_info['next_contract'] = {
                'symbol': next_contract.symbol,
                'code': next_contract.code,
                'expiry': next_contract.expiry_date
            }
        
        # Determine roll recommendations
        days_left = status['days_until_expiry']
        
        if days_left < 0:
            roll_info['must_roll'] = True
            roll_info['should_roll'] = True
            roll_info['recommendation'] = "URGENT: Contract has expired! Switch immediately."
            roll_info['warnings'].append("Trading expired contract may result in physical delivery")
            
        elif days_left <= 1:
            roll_info['must_roll'] = True
            roll_info['should_roll'] = True
            roll_info['recommendation'] = "CRITICAL: Contract expires within 1 day. Switch now."
            roll_info['warnings'].append("Last day to trade - extreme urgency")
            
        elif days_left <= 3:
            roll_info['should_roll'] = True
            roll_info['recommendation'] = "HIGH PRIORITY: Switch to next contract within 24 hours."
            roll_info['warnings'].append("Most volume has likely shifted to next contract")
            
        elif days_left <= 5:
            roll_info['should_roll'] = True
            roll_info['recommendation'] = "RECOMMENDED: Consider switching to next contract."
            roll_info['warnings'].append("Entering typical roll period")
            
        elif days_left <= 7:
            roll_info['recommendation'] = "MONITOR: Roll period approaching. Prepare for switch."
            roll_info['warnings'].append("Begin monitoring next contract volume")
            
        elif days_left <= 14:
            roll_info['recommendation'] = "WATCH: Two weeks until expiry. Plan roll strategy."
            
        else:
            roll_info['recommendation'] = "NORMAL: Continue trading current contract."
        
        # Suggest optimal roll date (Thursday of expiry week)
        if next_contract and days_left <= 14:
            expiry_date = current.expiry_datetime
            # Find Thursday before expiry week (typically best roll time)
            optimal_roll_date = expiry_date - timedelta(days=7 + (expiry_date.weekday() - 3) % 7)
            roll_info['roll_date_suggestion'] = optimal_roll_date.strftime("%Y-%m-%d")
        
        return roll_info
    
    def get_roll_timeline(self) -> List[Dict[str, any]]:
        """Get timeline of upcoming contract rolls"""
        timeline = []
        current_date = datetime.now()
        
        # Look ahead 6 months
        end_date = current_date + timedelta(days=180)
        
        for contract in ES_CONTRACTS_2024_2026.values():
            expiry = contract.expiry_datetime
            
            if current_date <= expiry <= end_date:
                # Calculate roll period dates
                roll_start = expiry - timedelta(days=5)
                optimal_roll = expiry - timedelta(days=7 + (expiry.weekday() - 3) % 7)
                
                timeline.append({
                    'contract': contract.symbol,
                    'code': contract.code,
                    'expiry_date': expiry.strftime("%Y-%m-%d"),
                    'roll_period_start': roll_start.strftime("%Y-%m-%d"),
                    'optimal_roll_date': optimal_roll.strftime("%Y-%m-%d"),
                    'days_until_expiry': (expiry - current_date).days,
                    'is_current': contract.symbol == self.current_contract.symbol
                })
        
        # Sort by expiry date
        timeline.sort(key=lambda x: x['expiry_date'])
        return timeline
    
    def record_roll(self, from_contract: str, to_contract: str, roll_date: str = None):
        """Record a contract roll in history"""
        if roll_date is None:
            roll_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        roll_record = {
            'from_contract': from_contract,
            'to_contract': to_contract,
            'roll_date': roll_date,
            'recorded_at': datetime.now().isoformat()
        }
        
        self.roll_history.append(roll_record)
        self._save_roll_history()
        
        return roll_record
    
    def get_roll_history(self) -> List[Dict[str, any]]:
        """Get historical roll records"""
        return self.roll_history.copy()
    
    def analyze_volume_shift(self, current_volume: int, next_volume: int) -> Dict[str, any]:
        """
        Analyze volume shift between contracts (if volume data available)
        
        Args:
            current_volume: Volume in current contract
            next_volume: Volume in next contract
            
        Returns:
            Analysis of volume shift and roll recommendation
        """
        total_volume = current_volume + next_volume
        
        if total_volume == 0:
            return {
                'status': 'NO_DATA',
                'recommendation': 'No volume data available',
                'current_ratio': 0,
                'next_ratio': 0
            }
        
        current_ratio = current_volume / total_volume
        next_ratio = next_volume / total_volume
        
        # Determine recommendation based on volume ratios
        if next_ratio > 0.6:
            recommendation = "ROLL NOW: Next contract has majority volume"
            status = "SHOULD_ROLL"
        elif next_ratio > 0.4:
            recommendation = "CONSIDER ROLLING: Volume is shifting"
            status = "MONITOR"
        else:
            recommendation = "STAY: Current contract still has majority volume"
            status = "STAY"
        
        return {
            'status': status,
            'recommendation': recommendation,
            'current_ratio': round(current_ratio, 3),
            'next_ratio': round(next_ratio, 3),
            'current_volume': current_volume,
            'next_volume': next_volume,
            'total_volume': total_volume
        }
    
    def generate_roll_alert(self) -> Optional[Dict[str, any]]:
        """Generate roll alert if action needed"""
        roll_status = self.check_roll_status()
        
        if roll_status['urgency'] in ['HIGH', 'CRITICAL']:
            return {
                'alert_type': 'CONTRACT_ROLL',
                'urgency': roll_status['urgency'],
                'message': roll_status['recommendation'],
                'current_contract': roll_status['current_contract'],
                'next_contract': roll_status['next_contract'],
                'days_left': roll_status['days_until_expiry'],
                'warnings': roll_status['warnings'],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _load_roll_history(self) -> List[Dict[str, any]]:
        """Load roll history from file"""
        try:
            if self.roll_history_file.exists():
                with open(self.roll_history_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return []
    
    def _save_roll_history(self):
        """Save roll history to file"""
        try:
            self.roll_history_file.parent.mkdir(exist_ok=True)
            with open(self.roll_history_file, 'w') as f:
                json.dump(self.roll_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save roll history: {e}")

def check_current_roll_status() -> Dict[str, any]:
    """Quick function to check current roll status"""
    roller = ContractRoller()
    return roller.check_roll_status()

def display_roll_status():
    """Display current roll status in formatted output"""
    roller = ContractRoller()
    status = roller.check_roll_status()
    
    print("ES Futures Contract Roll Status")
    print("=" * 40)
    print(f"Current Contract: {status['current_contract']} ({status['current_code']})")
    print(f"Days Until Expiry: {status['days_until_expiry']}")
    print(f"Status: {status['status']}")
    print(f"Urgency: {status['urgency']}")
    print(f"Recommendation: {status['recommendation']}")
    
    if status['next_contract']:
        next_contract = status['next_contract']
        print(f"Next Contract: {next_contract['symbol']} ({next_contract['code']})")
    
    if status['roll_date_suggestion']:
        print(f"Suggested Roll Date: {status['roll_date_suggestion']}")
    
    if status['warnings']:
        print("\nWarnings:")
        for warning in status['warnings']:
            print(f"  - {warning}")
    
    print("\nUpcoming Roll Timeline:")
    timeline = roller.get_roll_timeline()
    for item in timeline:
        current_indicator = " (CURRENT)" if item['is_current'] else ""
        print(f"  {item['contract']} expires {item['expiry_date']} "
              f"({item['days_until_expiry']} days){current_indicator}")

# Usage example and testing
if __name__ == "__main__":
    display_roll_status()
    
    # Test alert generation
    roller = ContractRoller()
    alert = roller.generate_roll_alert()
    
    if alert:
        print(f"\nROLL ALERT: {alert['urgency']}")
        print(f"Message: {alert['message']}")
    else:
        print("\nNo roll alerts at this time.")