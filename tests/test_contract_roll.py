#!/usr/bin/env python3
"""
Comprehensive ES Futures Contract Roll Testing
Tests contract configuration, rolling logic, and detection systems
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.contract_config import (
    ESContract, determine_current_contract, get_next_contract,
    get_contract_status, ES_CONTRACTS_2024_2026, 
    validate_contract_symbol, get_contract_by_symbol,
    get_third_friday, generate_next_contract
)
from utils.contract_roller import ContractRoller, check_current_roll_status


class TestESContractConfig(unittest.TestCase):
    """Test ES contract configuration and core functionality"""
    
    def test_contract_creation(self):
        """Test ESContract dataclass creation and properties"""
        contract = ESContract(
            symbol="ES 12-24",
            code="ESZ24", 
            expiry_date="2024-12-20",
            month_name="December",
            year=2024
        )
        
        self.assertEqual(contract.symbol, "ES 12-24")
        self.assertEqual(contract.code, "ESZ24")
        self.assertEqual(contract.expiry_date, "2024-12-20")
        self.assertEqual(contract.month_name, "December")
        self.assertEqual(contract.year, 2024)
        
        # Test datetime conversion
        expected_expiry = datetime(2024, 12, 20)
        self.assertEqual(contract.expiry_datetime, expected_expiry)
    
    def test_third_friday_calculation(self):
        """Test third Friday calculation for contract expiry"""
        # December 2024 - third Friday should be 20th
        third_friday_dec_2024 = get_third_friday(2024, 12)
        self.assertEqual(third_friday_dec_2024.day, 20)
        self.assertEqual(third_friday_dec_2024.weekday(), 4)  # Friday
        
        # March 2025 - third Friday should be 21st
        third_friday_mar_2025 = get_third_friday(2025, 3)
        self.assertEqual(third_friday_mar_2025.day, 21)
        self.assertEqual(third_friday_mar_2025.weekday(), 4)  # Friday
    
    def test_contract_validation(self):
        """Test contract symbol validation"""
        # Valid symbols
        self.assertTrue(validate_contract_symbol("ES 12-24"))
        self.assertTrue(validate_contract_symbol("ES 03-25"))
        self.assertTrue(validate_contract_symbol("ES 06-25"))
        self.assertTrue(validate_contract_symbol("ES 09-25"))
        
        # Invalid symbols
        self.assertFalse(validate_contract_symbol("ES 12-99"))  # Bad year
        self.assertFalse(validate_contract_symbol("ES 01-24"))  # Non-quarterly month
        self.assertFalse(validate_contract_symbol("ES12-24"))   # Missing space
        self.assertFalse(validate_contract_symbol("NQ 12-24"))  # Wrong instrument
        self.assertFalse(validate_contract_symbol(""))          # Empty
    
    def test_contract_lookup(self):
        """Test contract lookup by symbol"""
        # Should find existing contracts
        esz24 = get_contract_by_symbol("ES 12-24")
        self.assertIsNotNone(esz24)
        self.assertEqual(esz24.code, "ESZ24")
        
        # Should return None for non-existent contracts
        missing = get_contract_by_symbol("ES 01-24")
        self.assertIsNone(missing)
    
    def test_next_contract_logic(self):
        """Test getting next contract in sequence"""
        # Get December 2024 contract
        esz24 = ES_CONTRACTS_2024_2026["ESZ24"]
        
        # Next should be March 2025
        next_contract = get_next_contract(esz24)
        self.assertIsNotNone(next_contract)
        self.assertEqual(next_contract.symbol, "ES 03-25")
        self.assertEqual(next_contract.code, "ESH25")


class TestContractStatus(unittest.TestCase):
    """Test contract status and roll period detection"""
    
    def test_contract_status_active(self):
        """Test status for active contract (far from expiry)"""
        # Create a contract that expires far in the future
        future_date = datetime.now() + timedelta(days=60)
        future_contract = ESContract(
            symbol="ES 06-25",
            code="ESM25",
            expiry_date=future_date.strftime("%Y-%m-%d"),
            month_name="June",
            year=2025
        )
        
        status = get_contract_status(future_contract)
        self.assertEqual(status['status'], 'ACTIVE')
        self.assertEqual(status['urgency'], 'LOW')
        self.assertFalse(status['is_roll_period'])
    
    def test_contract_status_roll_period(self):
        """Test status for contract in roll period"""
        # Create contract expiring in 3 days
        near_expiry = datetime.now() + timedelta(days=3)
        roll_contract = ESContract(
            symbol="ES 12-24",
            code="ESZ24",
            expiry_date=near_expiry.strftime("%Y-%m-%d"),
            month_name="December",
            year=2024
        )
        
        status = get_contract_status(roll_contract)
        self.assertEqual(status['status'], 'ROLL_PERIOD')
        self.assertEqual(status['urgency'], 'HIGH')
        self.assertTrue(status['is_roll_period'])
    
    def test_contract_status_expired(self):
        """Test status for expired contract"""
        # Create contract that expired yesterday
        expired_date = datetime.now() - timedelta(days=1)
        expired_contract = ESContract(
            symbol="ES 09-24",
            code="ESU24",
            expiry_date=expired_date.strftime("%Y-%m-%d"),
            month_name="September",
            year=2024
        )
        
        status = get_contract_status(expired_contract)
        self.assertEqual(status['status'], 'EXPIRED')
        self.assertEqual(status['urgency'], 'CRITICAL')
        self.assertTrue(status['is_expired'])


class TestContractRoller(unittest.TestCase):
    """Test contract rolling utility"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.roller = ContractRoller()
    
    def test_roll_status_checking(self):
        """Test roll status checking functionality"""
        status = self.roller.check_roll_status()
        
        # Should return valid structure
        required_keys = [
            'current_contract', 'current_code', 'days_until_expiry',
            'status', 'urgency', 'should_roll', 'must_roll',
            'recommendation', 'warnings'
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
        
        # Status should be valid
        valid_statuses = ['ACTIVE', 'APPROACHING_EXPIRY', 'ROLL_PERIOD', 'EXPIRING_SOON', 'EXPIRED']
        self.assertIn(status['status'], valid_statuses)
        
        # Urgency should be valid
        valid_urgencies = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        self.assertIn(status['urgency'], valid_urgencies)
    
    def test_roll_timeline_generation(self):
        """Test roll timeline generation"""
        timeline = self.roller.get_roll_timeline()
        
        # Should return a list
        self.assertIsInstance(timeline, list)
        
        # Each item should have required structure
        if timeline:  # Only test if there are items
            item = timeline[0]
            required_keys = [
                'contract', 'code', 'expiry_date', 'roll_period_start',
                'optimal_roll_date', 'days_until_expiry', 'is_current'
            ]
            
            for key in required_keys:
                self.assertIn(key, item)
    
    def test_volume_analysis(self):
        """Test volume shift analysis"""
        # Test case 1: Current contract has majority volume
        analysis = self.roller.analyze_volume_shift(8000, 2000)
        self.assertEqual(analysis['status'], 'STAY')
        self.assertAlmostEqual(analysis['current_ratio'], 0.8, places=2)
        self.assertAlmostEqual(analysis['next_ratio'], 0.2, places=2)
        
        # Test case 2: Next contract has majority volume
        analysis = self.roller.analyze_volume_shift(3000, 7000)
        self.assertEqual(analysis['status'], 'SHOULD_ROLL')
        self.assertAlmostEqual(analysis['current_ratio'], 0.3, places=2)
        self.assertAlmostEqual(analysis['next_ratio'], 0.7, places=2)
        
        # Test case 3: Equal volume (transition period)
        analysis = self.roller.analyze_volume_shift(5000, 5000)
        self.assertEqual(analysis['status'], 'MONITOR')
        self.assertAlmostEqual(analysis['current_ratio'], 0.5, places=2)
        self.assertAlmostEqual(analysis['next_ratio'], 0.5, places=2)
        
        # Test case 4: No volume data
        analysis = self.roller.analyze_volume_shift(0, 0)
        self.assertEqual(analysis['status'], 'NO_DATA')
    
    def test_roll_history_management(self):
        """Test roll history recording and retrieval"""
        # Record a test roll
        initial_count = len(self.roller.get_roll_history())
        
        roll_record = self.roller.record_roll(
            from_contract="ES 09-24",
            to_contract="ES 12-24",
            roll_date="2024-09-15 10:30:00"
        )
        
        # Verify record structure
        self.assertEqual(roll_record['from_contract'], "ES 09-24")
        self.assertEqual(roll_record['to_contract'], "ES 12-24")
        self.assertEqual(roll_record['roll_date'], "2024-09-15 10:30:00")
        self.assertIn('recorded_at', roll_record)
        
        # Verify history was updated
        history = self.roller.get_roll_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIn(roll_record, history)


class TestCurrentContractDetermination(unittest.TestCase):
    """Test current contract determination logic"""
    
    def test_determine_current_contract_normal(self):
        """Test current contract determination under normal conditions"""
        # This will test against the actual current date
        current = determine_current_contract()
        
        # Should return a valid contract
        self.assertIsInstance(current, ESContract)
        self.assertIn(current.code, ES_CONTRACTS_2024_2026.keys())
    
    def test_determine_current_contract_historical(self):
        """Test current contract determination for historical dates"""
        # Test for a date in early September 2024 (before September expiry)
        sept_early = datetime(2024, 9, 1)
        contract_sept_early = determine_current_contract(sept_early)
        
        # Should return September 2024 contract (ESU24) as it hasn't expired yet
        self.assertEqual(contract_sept_early.code, "ESU24")
        
        # Test for date after September expiry but before December expiry
        oct_2024 = datetime(2024, 10, 1)
        contract_oct = determine_current_contract(oct_2024)
        
        # Should return December 2024 contract (ESZ24)
        self.assertEqual(contract_oct.code, "ESZ24")
        
        # Test for date close to December expiry
        near_expiry = datetime(2024, 12, 15)  # 5 days before expiry
        contract_near = determine_current_contract(near_expiry)
        
        # Should return March 2025 contract due to roll period
        self.assertEqual(contract_near.code, "ESH25")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete contract roll system"""
    
    def test_end_to_end_roll_detection(self):
        """Test complete roll detection workflow"""
        # Get current roll status
        status = check_current_roll_status()
        
        # Should return valid structure
        self.assertIsInstance(status, dict)
        self.assertIn('current_contract', status)
        self.assertIn('recommendation', status)
        
        # Test roller integration
        roller = ContractRoller()
        alert = roller.generate_roll_alert()
        
        # Alert may or may not exist depending on current date
        if alert:
            self.assertIn('alert_type', alert)
            self.assertIn('urgency', alert)
            self.assertIn('message', alert)
    
    def test_ninjascript_compatibility(self):
        """Test that contract data is compatible with NinjaScript format"""
        current = determine_current_contract()
        
        # Test symbol format (should be "ES MM-YY")
        self.assertTrue(current.symbol.startswith("ES "))
        self.assertEqual(len(current.symbol), 8)  # "ES 12-24" format
        
        # Test code format (should be "ESXYY")
        self.assertTrue(current.code.startswith("ES"))
        self.assertEqual(len(current.code), 5)  # "ESZ24" format
        
        # Test date format (should be ISO)
        datetime.fromisoformat(current.expiry_date)  # Should not raise exception


def run_contract_tests():
    """Run all contract roll tests"""
    print("Running ES Futures Contract Roll Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestESContractConfig,
        TestContractStatus,
        TestContractRoller,
        TestCurrentContractDetermination,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    # Run tests when called directly
    success = run_contract_tests()
    sys.exit(0 if success else 1)