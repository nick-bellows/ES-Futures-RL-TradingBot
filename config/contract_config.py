"""
ES Futures Contract Configuration
Handles quarterly contract rolls and expiry management for E-mini S&P 500 futures
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import calendar

@dataclass
class ESContract:
    """ES Futures contract details"""
    symbol: str          # e.g., "ES 12-24"
    code: str           # e.g., "ESZ24" 
    expiry_date: str    # ISO format "2024-12-20"
    month_name: str     # "December"
    year: int
    
    @property
    def expiry_datetime(self) -> datetime:
        """Get expiry date as datetime object"""
        return datetime.fromisoformat(self.expiry_date)
    
    @property
    def days_until_expiry(self) -> int:
        """Get days until contract expiry"""
        return (self.expiry_datetime - datetime.now()).days
    
    @property
    def is_in_roll_period(self) -> bool:
        """Check if we're in the roll period (5 days before expiry)"""
        return 0 <= self.days_until_expiry <= 5
    
    @property
    def is_expired(self) -> bool:
        """Check if contract has expired"""
        return self.days_until_expiry < 0

# ES futures quarterly contract definitions
ES_CONTRACTS_2024_2026 = {
    "ESU24": ESContract("ES 09-24", "ESU24", "2024-09-20", "September", 2024),
    "ESZ24": ESContract("ES 12-24", "ESZ24", "2024-12-20", "December", 2024),
    "ESH25": ESContract("ES 03-25", "ESH25", "2025-03-21", "March", 2025),
    "ESM25": ESContract("ES 06-25", "ESM25", "2025-06-20", "June", 2025),
    "ESU25": ESContract("ES 09-25", "ESU25", "2025-09-19", "September", 2025),
    "ESZ25": ESContract("ES 12-25", "ESZ25", "2025-12-19", "December", 2025),
    "ESH26": ESContract("ES 03-26", "ESH26", "2026-03-20", "March", 2026),
    "ESM26": ESContract("ES 06-26", "ESM26", "2026-06-19", "June", 2026),
    "ESU26": ESContract("ES 09-26", "ESU26", "2026-09-18", "September", 2026),
    "ESZ26": ESContract("ES 12-26", "ESZ26", "2026-12-18", "December", 2026),
}

# Month code mapping for futures
MONTH_CODES = {
    1: 'F',   # January
    2: 'G',   # February  
    3: 'H',   # March
    4: 'J',   # April
    5: 'K',   # May
    6: 'M',   # June
    7: 'N',   # July
    8: 'Q',   # August
    9: 'U',   # September
    10: 'V',  # October
    11: 'X',  # November
    12: 'Z'   # December
}

# ES quarterly months (March, June, September, December)
ES_QUARTERLY_MONTHS = [3, 6, 9, 12]

def get_third_friday(year: int, month: int) -> datetime:
    """
    Calculate the third Friday of a given month/year
    ES futures expire on the third Friday of quarterly months
    """
    # Find first day of month
    first_day = datetime(year, month, 1)
    
    # Find first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # Third Friday is 14 days later
    third_friday = first_friday + timedelta(days=14)
    
    return third_friday

def determine_current_contract(as_of_date: Optional[datetime] = None) -> ESContract:
    """
    Determine the current front month ES contract based on date
    
    Args:
        as_of_date: Date to check (defaults to today)
        
    Returns:
        ESContract object for current front month
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    
    # Check all available contracts
    for contract in ES_CONTRACTS_2024_2026.values():
        expiry = contract.expiry_datetime
        
        # If contract hasn't expired and we're not in roll period, use it
        if expiry > as_of_date:
            # Check if we should roll (within 5 days of expiry)
            days_until_expiry = (expiry - as_of_date).days
            
            if days_until_expiry > 5:
                return contract
            else:
                # In roll period - look for next contract
                continue
    
    # If we get here, find next available contract
    future_contracts = [
        contract for contract in ES_CONTRACTS_2024_2026.values()
        if contract.expiry_datetime > as_of_date
    ]
    
    if future_contracts:
        # Return earliest future contract
        return min(future_contracts, key=lambda c: c.expiry_datetime)
    
    # Fallback - create next quarter contract dynamically
    return generate_next_contract(as_of_date)

def get_next_contract(current_contract: ESContract) -> Optional[ESContract]:
    """Get the next quarterly contract after the current one"""
    current_expiry = current_contract.expiry_datetime
    
    # Find next contract
    future_contracts = [
        contract for contract in ES_CONTRACTS_2024_2026.values()
        if contract.expiry_datetime > current_expiry
    ]
    
    if future_contracts:
        return min(future_contracts, key=lambda c: c.expiry_datetime)
    
    return None

def generate_next_contract(from_date: datetime) -> ESContract:
    """Generate next ES contract dynamically if not in predefined list"""
    year = from_date.year
    
    # Find next quarterly month
    for month in ES_QUARTERLY_MONTHS:
        if month > from_date.month:
            next_month = month
            next_year = year
            break
    else:
        # Next year
        next_month = ES_QUARTERLY_MONTHS[0]  # March
        next_year = year + 1
    
    # Calculate expiry (third Friday)
    expiry_date = get_third_friday(next_year, next_month)
    
    # Generate contract details
    month_code = MONTH_CODES[next_month]
    year_code = str(next_year)[-2:]  # Last 2 digits
    
    code = f"ES{month_code}{year_code}"
    symbol = f"ES {next_month:02d}-{str(next_year)[-2:]}"
    month_name = calendar.month_name[next_month]
    
    return ESContract(
        symbol=symbol,
        code=code,
        expiry_date=expiry_date.strftime("%Y-%m-%d"),
        month_name=month_name,
        year=next_year
    )

def get_contract_status(contract: ESContract) -> Dict[str, any]:
    """Get comprehensive status of a contract"""
    now = datetime.now()
    expiry = contract.expiry_datetime
    days_until_expiry = (expiry - now).days
    
    # Determine status
    if days_until_expiry < 0:
        status = "EXPIRED"
        urgency = "CRITICAL"
    elif days_until_expiry <= 1:
        status = "EXPIRING_SOON"
        urgency = "CRITICAL"
    elif days_until_expiry <= 5:
        status = "ROLL_PERIOD"
        urgency = "HIGH"
    elif days_until_expiry <= 14:
        status = "APPROACHING_EXPIRY"
        urgency = "MEDIUM"
    else:
        status = "ACTIVE"
        urgency = "LOW"
    
    return {
        'contract': contract,
        'status': status,
        'urgency': urgency,
        'days_until_expiry': days_until_expiry,
        'expiry_date': expiry.strftime("%Y-%m-%d"),
        'is_roll_period': contract.is_in_roll_period,
        'is_expired': contract.is_expired,
        'next_contract': get_next_contract(contract)
    }

# Current configuration (Updated for December 2024)
CURRENT_CONTRACT = determine_current_contract()
CONTRACT_SYMBOL = CURRENT_CONTRACT.symbol
CONTRACT_CODE = CURRENT_CONTRACT.code
EXPIRY_DATE = CURRENT_CONTRACT.expiry_date
NEXT_CONTRACT = get_next_contract(CURRENT_CONTRACT)

def display_contract_info(contract: ESContract = None) -> str:
    """Display formatted contract information"""
    if contract is None:
        contract = CURRENT_CONTRACT
    
    status = get_contract_status(contract)
    next_contract = status['next_contract']
    
    info = [
        f"Current Contract: {contract.symbol} ({contract.code})",
        f"Expiry Date: {contract.expiry_date} ({contract.month_name} {contract.year})",
        f"Days Until Expiry: {status['days_until_expiry']}",
        f"Status: {status['status']} (Urgency: {status['urgency']})",
    ]
    
    if next_contract:
        info.append(f"Next Contract: {next_contract.symbol} ({next_contract.code})")
    
    if status['is_roll_period']:
        info.append("WARNING: In roll period - consider switching to next contract")
    
    return "\n".join(info)

# Contract validation
def validate_contract_symbol(symbol: str) -> bool:
    """Validate if a contract symbol is properly formatted"""
    # ES 12-24 format
    if symbol.startswith("ES ") and len(symbol) == 8:
        try:
            month_part = symbol[3:5]
            year_part = symbol[6:8]
            month = int(month_part)
            year = int(f"20{year_part}")
            return month in ES_QUARTERLY_MONTHS and 2024 <= year <= 2030
        except ValueError:
            return False
    
    return False

def get_contract_by_symbol(symbol: str) -> Optional[ESContract]:
    """Get contract object by symbol"""
    for contract in ES_CONTRACTS_2024_2026.values():
        if contract.symbol == symbol:
            return contract
    return None

# Usage example and testing
if __name__ == "__main__":
    print("ES Futures Contract Configuration")
    print("=" * 40)
    
    current = determine_current_contract()
    print(f"Current front month: {current.symbol}")
    print(f"Contract code: {current.code}")
    print(f"Expiry: {current.expiry_date}")
    print(f"Days until expiry: {current.days_until_expiry}")
    print(f"In roll period: {current.is_in_roll_period}")
    print()
    
    print("Contract Status:")
    print(display_contract_info(current))
    print()
    
    print("All Available Contracts:")
    for code, contract in ES_CONTRACTS_2024_2026.items():
        status = get_contract_status(contract)
        print(f"{code}: {contract.symbol} | {status['status']} | {status['days_until_expiry']} days")
    
    print(f"\nNext contract: {get_next_contract(current)}")