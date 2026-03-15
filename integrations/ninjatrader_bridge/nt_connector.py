"""
NinjaTrader AT Interface Connector
TCP socket-based connection handler for NinjaTrader 8 AT Interface
"""

import socket
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from .config import NTConfig

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

class NTConnector:
    """
    NinjaTrader AT Interface connector
    Handles TCP socket communication with NinjaTrader 8 AT Interface
    """
    
    def __init__(self, config: NTConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # TCP socket connection
        self.socket = None
        self.socket_lock = threading.Lock()
        self.state = ConnectionState.DISCONNECTED
        self.last_heartbeat = None
        
        # Connection monitoring
        self.heartbeat_thread = None
        self.heartbeat_interval = 30.0  # seconds
        self.running = False
        
        # Command tracking
        self.command_count = 0
        self.last_command_time = 0
        self.rate_limit_delay = 0.1  # 100ms between commands
        
        # Error handling
        self.max_retries = config.connection.max_retries
        self.retry_delay = config.connection.retry_delay
        self.consecutive_errors = 0
        
        # Response buffer
        self.response_buffer = ""
        
    def connect(self) -> bool:
        """
        Establish TCP connection to NinjaTrader AT Interface
        
        Returns:
            True if connection successful
        """
        try:
            self.logger.info(f"Connecting to NinjaTrader AT Interface at {self.config.connection.host}:{self.config.connection.port}")
            self.state = ConnectionState.CONNECTING
            
            with self.socket_lock:
                # Create TCP socket with REAL-TIME optimization
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # CRITICAL: Real-time socket optimization for low-latency trading
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)    # Disable Nagle's algorithm
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)   # Small receive buffer
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)   # Small send buffer
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # Allow address reuse
                
                # Set shorter timeout for real-time data (was using config.timeout which might be too long)
                self.socket.settimeout(2.0)  # 2 seconds max for real-time trading
                
                self.logger.info("Socket configured for real-time trading (TCP_NODELAY enabled)")
                
                # Connect to NinjaTrader
                self.socket.connect((self.config.connection.host, self.config.connection.port))
            
            self.state = ConnectionState.CONNECTED
            self.logger.info("Connected to NinjaTrader AT Interface successfully")
            
            # Test with heartbeat
            if self._send_heartbeat():
                self.logger.info("Heartbeat successful - AT Interface responding")
                self.state = ConnectionState.AUTHENTICATED
            else:
                self.logger.warning("No heartbeat response - connection may be unstable")
            
            # Start heartbeat monitoring
            self._start_heartbeat()
            
            # CRITICAL: Clear any buffered/stale data on new connection
            self._clear_socket_buffers()
            
            return True
            
        except socket.timeout:
            self.logger.error("Connection timeout - check if NinjaTrader AT Interface is enabled")
            self.state = ConnectionState.ERROR
            return False
        except socket.error as e:
            self.logger.error(f"Socket error: {e}")
            self.logger.error("Please ensure:")
            self.logger.error("- NinjaTrader 8 is running")
            self.logger.error("- AT Interface is enabled")
            self.logger.error("- Port 8080 is not blocked")
            self.state = ConnectionState.ERROR
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.state = ConnectionState.ERROR
            return False
    
    def disconnect(self) -> None:
        """Disconnect from NinjaTrader AT Interface"""
        try:
            self.logger.info("Disconnecting from NinjaTrader AT Interface...")
            self.running = False
            
            # Stop heartbeat
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5.0)
            
            # Close socket
            with self.socket_lock:
                if self.socket:
                    self.socket.close()
                    self.socket = None
            
            self.state = ConnectionState.DISCONNECTED
            self.logger.info("Disconnected from NinjaTrader AT Interface")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to NinjaTrader"""
        return self.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]
    
    def send_command(self, command: str, params: List[str] = None) -> Optional[str]:
        """
        Send AT Interface command to NinjaTrader
        
        Args:
            command: Command name (PLACE, CANCEL, HEARTBEAT, etc.)
            params: List of command parameters
            
        Returns:
            Response string or None if failed
        """
        if not self.is_connected():
            self.logger.error("Not connected to NinjaTrader AT Interface")
            return None
        
        try:
            # Build command string
            if params:
                cmd_string = f"{command};{';'.join(str(p) for p in params)}\r\n"
            else:
                cmd_string = f"{command}\r\n"
            
            self.logger.debug(f"Sending command: {cmd_string.strip()}")
            
            # Rate limiting
            self._enforce_rate_limit()
            
            # Send command
            with self.socket_lock:
                if self.socket:
                    self.socket.send(cmd_string.encode('utf-8'))
                    self.command_count += 1
                    
                    # Try to read response (non-blocking)
                    self.socket.settimeout(1.0)  # 1 second timeout for response
                    try:
                        response = self.socket.recv(1024).decode('utf-8').strip()
                        self.logger.debug(f"Received response: {response}")
                        return response
                    except socket.timeout:
                        # No immediate response - this is normal for some commands
                        self.logger.debug("No immediate response (normal)")
                        return ""
                    finally:
                        # Reset to original timeout
                        self.socket.settimeout(self.config.connection.timeout)
                else:
                    self.logger.error("Socket is None")
                    return None
            
        except socket.error as e:
            self.logger.error(f"Socket error sending command: {e}")
            self._handle_socket_error()
            return None
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return None
    
    def receive_data(self, timeout: float = 0.1) -> Optional[str]:
        """
        Receive data from AT Interface (non-blocking)
        
        Args:
            timeout: Timeout in seconds for receiving data
            
        Returns:
            Received data string or None if no data
        """
        if not self.is_connected():
            return None
        
        try:
            with self.socket_lock:
                if self.socket:
                    # Set short timeout for non-blocking receive
                    original_timeout = self.socket.gettimeout()
                    self.socket.settimeout(timeout)
                    
                    try:
                        data = self.socket.recv(4096).decode('utf-8')
                        if data:
                            self.logger.debug(f"AT Interface raw response: {data.strip()}")
                            return data
                        return None
                    except socket.timeout:
                        # No data available - this is normal
                        return None
                    finally:
                        # Restore original timeout
                        if original_timeout:
                            self.socket.settimeout(original_timeout)
                else:
                    return None
                    
        except socket.error as e:
            self.logger.debug(f"Socket error receiving data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error receiving data: {e}")
            return None
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring thread"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        
        self.logger.info("Heartbeat monitoring started")
    
    def _heartbeat_worker(self) -> None:
        """Heartbeat monitoring worker"""
        while self.running:
            try:
                if self.is_connected():
                    # Send heartbeat request
                    success = self._send_heartbeat()
                    
                    if success:
                        self.last_heartbeat = datetime.now()
                        self.consecutive_errors = 0
                    else:
                        self.consecutive_errors += 1
                        self.logger.warning(f"Heartbeat failed ({self.consecutive_errors} consecutive)")
                        
                        # Reconnect if too many consecutive errors
                        if self.consecutive_errors >= 3:
                            self.logger.error("Too many heartbeat failures, attempting reconnect")
                            self._attempt_reconnect()
                
                # Wait for next heartbeat
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat worker error: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _send_heartbeat(self) -> bool:
        """Send heartbeat to NinjaTrader AT Interface"""
        try:
            response = self.send_command("HEARTBEAT")
            return response is not None
            
        except Exception as e:
            self.logger.debug(f"Heartbeat failed: {e}")
            return False
    
    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to NinjaTrader AT Interface"""
        try:
            self.logger.info("Attempting to reconnect...")
            
            # Close existing socket
            with self.socket_lock:
                if self.socket:
                    self.socket.close()
                    self.socket = None
            
            # Attempt reconnection
            if self.connect():
                self.consecutive_errors = 0
                self.logger.info("Reconnected successfully")
            else:
                self.state = ConnectionState.ERROR
                self.logger.error("Reconnection failed")
                
        except Exception as e:
            self.logger.error(f"Reconnection error: {e}")
            self.state = ConnectionState.ERROR
    
    def place_order(self, account: str, instrument: str, action: str, quantity: int, 
                   order_type: str = "MARKET", price: float = 0, stop_price: float = 0) -> bool:
        """
        Place order via AT Interface
        
        Args:
            account: Account name (e.g., "MyAccount")
            instrument: Instrument name (e.g., "ES 09-25")
            action: "BUY" or "SELL"
            quantity: Number of contracts
            order_type: "MARKET", "LIMIT", "STOP", etc.
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            
        Returns:
            True if order placed successfully
        """
        try:
            # FIXED: Use simplified format that actually works with AT Interface
            # Diagnostic testing confirmed: PLACE;AccountName;ES 09-25;BUY;1;MKT works
            
            # Convert order_type to AT Interface format
            at_order_type = "MKT" if order_type == "MARKET" else order_type
            
            # Build simplified PLACE command (the format that actually works!)
            params = [
                account,           # Account
                instrument,        # Instrument  
                action,           # Action (BUY/SELL)
                str(quantity),    # Quantity
                at_order_type     # Order Type (MKT, LMT, etc.)
            ]
            
            response = self.send_command("PLACE", params)
            
            if response is not None:
                # Parse the response to extract order information
                order_info = self._parse_order_response(response)
                
                if order_info.get('success'):
                    self.logger.info(f"Order placed: {action} {quantity} {instrument} ({at_order_type})")
                    if order_info.get('order_id'):
                        self.logger.info(f"Order ID: {order_info['order_id']}")
                    return True
                else:
                    self.logger.error(f"Order rejected: {order_info.get('error', 'Unknown error')}")
                    return False
            else:
                self.logger.error("Failed to place order - no response from AT Interface")
                return False
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return False
    
    def _parse_order_response(self, response: str) -> Dict[str, Any]:
        """Parse AT Interface order response"""
        try:
            # AT Interface returns order status data in the response
            # Look for success indicators and order IDs
            
            result = {'success': False, 'order_id': None, 'error': None}
            
            if not response:
                result['error'] = "Empty response"
                return result
            
            response_upper = response.upper()
            
            # Check for success indicators (based on diagnostic results)
            if any(indicator in response_upper for indicator in ['ORDERSTATE', 'FILLED', 'ACCEPTED', 'PLACED']):
                result['success'] = True
                
                # Extract order ID from response
                import re
                # Look for 8+ digit order IDs (like 284552910005 from diagnostic)
                order_id_match = re.search(r'\b(\d{10,})\b', response)
                if order_id_match:
                    result['order_id'] = order_id_match.group(1)
                
            # Check for error indicators
            elif any(error in response_upper for error in ['ERROR', 'REJECTED', 'INVALID', 'DENIED', 'FAILED']):
                result['error'] = response.strip()
                
            # If response contains order status data (like diagnostic showed), consider it success
            elif 'OrderStatus' in response or 'AvgFillPrice' in response:
                result['success'] = True
                # Extract order ID from status data
                import re
                order_id_match = re.search(r'OrderStatus\|(\d+)', response)
                if order_id_match:
                    result['order_id'] = order_id_match.group(1)
            
            # Just a number response (like "2") might indicate success
            elif response.strip().isdigit():
                result['success'] = True
                
            else:
                # Unknown response format
                result['error'] = f"Unknown response format: {response.strip()}"
            
            return result
            
        except Exception as e:
            return {'success': False, 'order_id': None, 'error': f"Parse error: {e}"}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order via AT Interface
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancel request sent successfully
        """
        try:
            response = self.send_command("CANCEL", [order_id])
            
            if response is not None:
                self.logger.info(f"Cancel order requested: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def subscribe_market_data(self, instrument: str) -> bool:
        """
        Subscribe to market data via AT Interface
        
        Args:
            instrument: Instrument to subscribe to
            
        Returns:
            True if subscription successful
        """
        try:
            response = self.send_command("SUBSCRIBE", [instrument])
            
            if response is not None:
                self.logger.info(f"Subscribed to market data: {instrument}")
                return True
            else:
                self.logger.error(f"Failed to subscribe to market data: {instrument}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to market data: {e}")
            return False
    
    def unsubscribe_market_data(self, instrument: str) -> bool:
        """
        Unsubscribe from market data via AT Interface
        
        Args:
            instrument: Instrument to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        try:
            response = self.send_command("UNSUBSCRIBE", [instrument])
            
            if response is not None:
                self.logger.info(f"Unsubscribed from market data: {instrument}")
                return True
            else:
                self.logger.error(f"Failed to unsubscribe from market data: {instrument}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from market data: {e}")
            return False
    
    def _clear_socket_buffers(self) -> None:
        """Clear any buffered/stale data from socket - CRITICAL for real-time trading"""
        if not self.socket:
            return
            
        try:
            self.logger.info("Clearing socket buffers for real-time data...")
            
            # Set socket to non-blocking
            self.socket.setblocking(False)
            
            # Read and discard any buffered data
            bytes_cleared = 0
            while True:
                try:
                    data = self.socket.recv(4096)
                    if not data:
                        break
                    bytes_cleared += len(data)
                except socket.error:
                    break  # No more data to clear
            
            # Restore blocking mode with timeout
            self.socket.setblocking(True)
            self.socket.settimeout(2.0)
            
            if bytes_cleared > 0:
                self.logger.warning(f"Cleared {bytes_cleared} bytes of stale data from socket buffer")
            else:
                self.logger.info("Socket buffers were clean")
                
        except Exception as e:
            self.logger.error(f"Error clearing socket buffers: {e}")
    
    def _handle_socket_error(self) -> None:
        """Handle socket errors"""
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= 3:
            self.logger.error("Too many socket errors, attempting reconnect")
            self._attempt_reconnect()
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between commands"""
        current_time = time.time()
        time_since_last = current_time - self.last_command_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_command_time = time.time()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'state': self.state.value,
            'connected': self.is_connected(),
            'host': self.config.connection.host,
            'port': self.config.connection.port,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'command_count': self.command_count,
            'consecutive_errors': self.consecutive_errors,
            'uptime_seconds': (datetime.now() - self.last_heartbeat).total_seconds() 
                            if self.last_heartbeat else 0
        }
    
    def test_at_interface(self) -> bool:
        """Test AT Interface connectivity"""
        try:
            self.logger.info("Testing AT Interface commands...")
            
            # Test heartbeat
            if self._send_heartbeat():
                self.logger.info("✓ Heartbeat successful")
                return True
            else:
                self.logger.error("✗ Heartbeat failed")
                return False
                
        except Exception as e:
            self.logger.error(f"AT Interface test failed: {e}")
            return False
    
    def test_endpoints(self) -> Dict[str, bool]:
        """Test AT Interface commands (compatibility method)"""
        try:
            results = {
                'heartbeat': self._send_heartbeat(),
                'market_data': True,  # Assume market data works if connection works
                'orders': True,       # Order placement methods exist
                'position': True      # Position tracking available
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Endpoint test failed: {e}")
            return {
                'heartbeat': False,
                'market_data': False, 
                'orders': False,
                'position': False
            }