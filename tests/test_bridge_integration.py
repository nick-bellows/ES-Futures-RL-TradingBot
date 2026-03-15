"""
Bridge Integration Test (Project-based)
Complete end-to-end test of the NinjaTrader bridge system
"""

import sys
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.bridge_config import BRIDGE_CONFIG
from ninjatrader_bridge.market_data_bridge import ProjectMarketDataBridge
from ninjatrader_bridge.signal_writer import ProjectSignalWriter, TradingAction
from ninjatrader_bridge.execution_monitor import ProjectExecutionMonitor

class BridgeIntegrationTester:
    """
    Complete integration test for project-based bridge system
    Tests all components working together
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize bridge components
        self.market_bridge = ProjectMarketDataBridge()
        self.signal_writer = ProjectSignalWriter()
        self.execution_monitor = ProjectExecutionMonitor()
        
        # Test results
        self.test_results = {
            'setup_check': False,
            'market_data_test': False,
            'signal_writing_test': False,
            'execution_monitoring_test': False,
            'integration_test': False,
            'all_tests_passed': False
        }
        
        self.logger.info("Bridge Integration Tester initialized")
    
    def run_complete_test(self) -> dict:
        """Run complete integration test suite"""
        print("=" * 60)
        print("PROJECT-BASED BRIDGE INTEGRATION TEST")
        print("=" * 60)
        
        # Test 1: Setup and configuration
        print("\n1. Testing bridge setup and configuration...")
        self.test_results['setup_check'] = self._test_setup()
        print(f"   Setup check: {'PASS' if self.test_results['setup_check'] else 'FAIL'}")
        
        # Test 2: Market data bridge
        print("\n2. Testing market data bridge...")
        self.test_results['market_data_test'] = self._test_market_data()
        print(f"   Market data: {'PASS' if self.test_results['market_data_test'] else 'FAIL'}")
        
        # Test 3: Signal writer
        print("\n3. Testing signal writer...")
        self.test_results['signal_writing_test'] = self._test_signal_writer()
        print(f"   Signal writer: {'PASS' if self.test_results['signal_writing_test'] else 'FAIL'}")
        
        # Test 4: Execution monitor
        print("\n4. Testing execution monitor...")
        self.test_results['execution_monitoring_test'] = self._test_execution_monitor()
        print(f"   Execution monitor: {'PASS' if self.test_results['execution_monitoring_test'] else 'FAIL'}")
        
        # Test 5: Integration test
        print("\n5. Testing full integration...")
        self.test_results['integration_test'] = self._test_integration()
        print(f"   Integration: {'PASS' if self.test_results['integration_test'] else 'FAIL'}")
        
        # Final results
        self.test_results['all_tests_passed'] = all(self.test_results.values())
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:25}: {status}")
        
        print("\n" + "=" * 60)
        if self.test_results['all_tests_passed']:
            print("ALL TESTS PASSED - Bridge system ready for production")
        else:
            print("SOME TESTS FAILED - Review setup and configuration")
        print("=" * 60)
        
        return self.test_results
    
    def _test_setup(self) -> bool:
        """Test bridge setup and configuration"""
        try:
            # Check project paths
            bridge_dir = Path(BRIDGE_CONFIG['bridge_directory'])
            if not bridge_dir.exists():
                print(f"   ERROR: Bridge directory missing: {bridge_dir}")
                print("   Run: python utils/setup_bridge.py")
                return False
            
            # Check configuration
            required_files = [
                BRIDGE_CONFIG['market_data_file'],
                BRIDGE_CONFIG['signals_file'],
                BRIDGE_CONFIG['status_file'],
                BRIDGE_CONFIG['execution_log_file']
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).parent.exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"   ERROR: Missing directories for files: {missing_files}")
                return False
            
            # Check configuration values
            if BRIDGE_CONFIG['max_data_age_seconds'] <= 0:
                print("   ERROR: Invalid max_data_age_seconds configuration")
                return False
            
            print("   Bridge directory structure: OK")
            print("   Configuration values: OK")
            return True
            
        except Exception as e:
            print(f"   ERROR in setup test: {e}")
            return False
    
    def _test_market_data(self) -> bool:
        """Test market data bridge functionality"""
        try:
            # Test emergency check
            emergency_check = self.market_bridge.emergency_check()
            print(f"   Market data status: {emergency_check['status']}")
            
            if emergency_check['status'] == 'CRITICAL':
                print("   ERROR: Market data not available")
                print("   Ensure NinjaScript ESDataBridge is running")
                return False
            
            # Test data validation
            test_prices = [6595.50, 6584.75, 6600.00]
            for price in test_prices:
                if not self.market_bridge.validate_price(price):
                    print(f"   ERROR: Price validation failed for {price}")
                    return False
            
            # Test statistics
            stats = self.market_bridge.get_statistics()
            if not isinstance(stats, dict):
                print("   ERROR: Statistics not returned as dictionary")
                return False
            
            print("   Emergency check: OK")
            print("   Price validation: OK")
            print("   Statistics: OK")
            return True
            
        except Exception as e:
            print(f"   ERROR in market data test: {e}")
            return False
    
    def _test_signal_writer(self) -> bool:
        """Test signal writer functionality"""
        try:
            # Clear any existing signals
            self.signal_writer.clear_signals()
            
            # Test different signal types
            test_price = 6595.50
            
            # Test BUY signal
            if not self.signal_writer.write_buy_signal(test_price, 0.4523):
                print("   ERROR: Failed to write BUY signal")
                return False
            
            time.sleep(1.1)  # Wait for minimum interval
            
            # Test SELL signal
            if not self.signal_writer.write_sell_signal(test_price - 2.5, 0.3891):
                print("   ERROR: Failed to write SELL signal")
                return False
            
            time.sleep(1.1)
            
            # Test FLAT signal
            if not self.signal_writer.write_flat_signal(test_price - 1.0):
                print("   ERROR: Failed to write FLAT signal")
                return False
            
            # Test signal reading
            last_signal = self.signal_writer.get_last_signal()
            if not last_signal or 'FLAT' not in last_signal:
                print("   ERROR: Last signal not readable or incorrect")
                return False
            
            # Test statistics
            stats = self.signal_writer.get_signal_stats()
            if stats['signal_count'] < 3:
                print(f"   ERROR: Expected 3+ signals, got {stats['signal_count']}")
                return False
            
            print("   BUY signal: OK")
            print("   SELL signal: OK")
            print("   FLAT signal: OK")
            print("   Signal reading: OK")
            print("   Statistics: OK")
            return True
            
        except Exception as e:
            print(f"   ERROR in signal writer test: {e}")
            return False
    
    def _test_execution_monitor(self) -> bool:
        """Test execution monitor functionality"""
        try:
            # Test statistics
            stats = self.execution_monitor.get_statistics()
            if not isinstance(stats, dict):
                print("   ERROR: Statistics not returned as dictionary")
                return False
            
            # Test NinjaScript status
            status = self.execution_monitor.get_ninjascript_status()
            print(f"   NinjaScript status: {status['status']}")
            
            # Test order summary
            summary = self.execution_monitor.get_order_summary()
            if not isinstance(summary, dict):
                print("   ERROR: Order summary not returned as dictionary")
                return False
            
            # Test error summary
            errors = self.execution_monitor.get_error_summary()
            if not isinstance(errors, dict):
                print("   ERROR: Error summary not returned as dictionary")
                return False
            
            print("   Statistics: OK")
            print("   Order summary: OK")
            print("   Error summary: OK")
            return True
            
        except Exception as e:
            print(f"   ERROR in execution monitor test: {e}")
            return False
    
    def _test_integration(self) -> bool:
        """Test full integration of all components"""
        try:
            print("   Testing integrated workflow...")
            
            # Step 1: Check if NinjaScript components are available
            market_check = self.market_bridge.emergency_check()
            if market_check['status'] in ['CRITICAL', 'ERROR']:
                print("   INTEGRATION NOTE: NinjaScript not running - testing file operations only")
                
                # Test file-based integration without live data
                test_price = 6595.50
                
                # Clear any existing signals first
                self.signal_writer.clear_signals()
                time.sleep(0.1)
                
                # Write a test signal
                success = self.signal_writer.write_buy_signal(test_price, 0.4523)
                if not success:
                    print("   ERROR: Integration signal write failed")
                    return False
                
                # Verify signal file exists and is readable
                signal_file = Path(BRIDGE_CONFIG['signals_file'])
                if not signal_file.exists():
                    print("   ERROR: Signal file not created during integration")
                    return False
                
                # Read back the signal
                last_signal = self.signal_writer.get_last_signal()
                if not last_signal or str(test_price) not in last_signal:
                    print("   ERROR: Signal not readable or incorrect in integration test")
                    return False
                
                print("   File-based integration: OK")
                return True
            
            else:
                # Full integration test with live NinjaScript
                print("   Live NinjaScript detected - running full integration test")
                
                # Start market data monitoring
                if not self.market_bridge.start_monitoring():
                    print("   ERROR: Could not start market data monitoring")
                    return False
                
                # Wait for market data
                time.sleep(2.0)
                current_price = self.market_bridge.get_current_price()
                
                if current_price <= 0:
                    print("   ERROR: No market data received during integration test")
                    self.market_bridge.stop_monitoring()
                    return False
                
                # Write signal based on live price
                success = self.signal_writer.write_buy_signal(current_price, 0.4523)
                if not success:
                    print("   ERROR: Could not write signal with live price")
                    self.market_bridge.stop_monitoring()
                    return False
                
                # Stop monitoring
                self.market_bridge.stop_monitoring()
                
                print("   Live integration: OK")
                return True
            
        except Exception as e:
            print(f"   ERROR in integration test: {e}")
            return False
    
    def generate_deployment_report(self) -> str:
        """Generate deployment report"""
        report = []
        report.append("PROJECT-BASED BRIDGE DEPLOYMENT REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Project Root: {PROJECT_ROOT}")
        report.append("")
        
        # Bridge configuration
        report.append("BRIDGE CONFIGURATION:")
        for key, value in BRIDGE_CONFIG.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # Test results
        report.append("TEST RESULTS:")
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            report.append(f"  {test_name}: {status}")
        report.append("")
        
        # Next steps
        report.append("DEPLOYMENT STEPS:")
        report.append("1. Copy NinjaScript files to NinjaTrader:")
        report.append("   - Copy UpdatedESDataBridge.cs to Documents\\NinjaTrader 8\\bin\\Custom\\Indicators\\")
        report.append("   - Copy UpdatedESSignalExecutor.cs to Documents\\NinjaTrader 8\\bin\\Custom\\Strategies\\")
        report.append("2. Compile NinjaScript files (F5 in NinjaScript Editor)")
        report.append("3. Add ESDataBridge indicator to ES futures chart")
        report.append("4. Add ESSignalExecutor strategy to same chart")
        report.append("5. Verify bridge files are created in D:\\QC_TradingBot_v3\\data\\bridge\\")
        report.append("6. Run Python bot with project-based bridge")
        report.append("")
        
        # Troubleshooting
        if not self.test_results['all_tests_passed']:
            report.append("TROUBLESHOOTING:")
            if not self.test_results['setup_check']:
                report.append("- Run: python utils/setup_bridge.py")
            if not self.test_results['market_data_test']:
                report.append("- Start NinjaTrader and connect to data feed")
                report.append("- Add ESDataBridge indicator to ES chart")
            report.append("")
        
        return "\n".join(report)

# Run integration test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tester = BridgeIntegrationTester()
    results = tester.run_complete_test()
    
    # Generate and save deployment report
    report = tester.generate_deployment_report()
    
    report_file = PROJECT_ROOT / "tests" / "deployment_report.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDeployment report saved to: {report_file}")
    
    if results['all_tests_passed']:
        print("\nBridge system ready for production deployment!")
    else:
        print("\nReview failed tests before deployment.")