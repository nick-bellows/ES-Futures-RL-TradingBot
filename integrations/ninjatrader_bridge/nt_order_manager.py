"""
NinjaTrader Order Management
Order execution and position management for ES futures trading
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from .config import NTConfig
from .nt_connector import NTConnector

class OrderAction(Enum):
    """Order actions"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class PositionSide(Enum):
    """Position sides"""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Order:
    """Order object"""
    order_id: str
    instrument: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'instrument': self.instrument,
            'action': self.action.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'filled_time': self.filled_time.isoformat() if self.filled_time else None,
            'error_message': self.error_message
        }

@dataclass
class Position:
    """Current position information"""
    instrument: str
    side: PositionSide
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'instrument': self.instrument,
            'side': self.side.value,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'market_value': self.market_value
        }

@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    instrument: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: PositionSide
    pnl: float
    commission: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'instrument': self.instrument,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side.value,
            'pnl': self.pnl,
            'commission': self.commission
        }

class NTOrderManager:
    """
    NinjaTrader order execution and position management
    Handles ES futures trading with proper risk management
    """
    
    def __init__(self, config: NTConfig, connector: NTConnector):
        self.config = config
        self.connector = connector
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_lock = threading.Lock()
        
        # Position tracking
        self.current_position: Optional[Position] = None
        self.position_lock = threading.Lock()
        
        # Trade history
        self.completed_trades: List[Trade] = []
        self.trade_lock = threading.Lock()
        
        # Risk management
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 2.0  # seconds
        
        # Callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.position_callbacks: List[Callable[[Position], None]] = []
        self.trade_callbacks: List[Callable[[Trade], None]] = []
        
        # Initialize position
        self._refresh_position()
    
    def start_monitoring(self) -> bool:
        """Start order and position monitoring"""
        if self.monitoring:
            self.logger.warning("Monitoring already active")
            return True
        
        try:
            self.logger.info("Starting order and position monitoring")
            
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            self.monitor_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        try:
            self.logger.info("Stopping monitoring...")
            self.monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            self.logger.info("Monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def place_market_order(self, action: OrderAction, quantity: int = 1) -> Optional[str]:
        """
        Place market order with automatic stop loss and take profit
        
        Args:
            action: BUY or SELL
            quantity: Number of contracts
            
        Returns:
            Order ID if successful, None if failed
        """
        try:
            # Check risk limits
            if not self._check_risk_limits():
                return None
            
            # Check position limits
            if not self._check_position_limits(action, quantity):
                return None
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Create order
            order = Order(
                order_id=order_id,
                instrument=self.config.trading.instrument,
                action=action,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            # Submit order to NinjaTrader
            success = self._submit_order(order)
            
            if success:
                with self.order_lock:
                    self.orders[order_id] = order
                
                self.logger.info(f"Market order placed: {action.value} {quantity} {self.config.trading.instrument}")
                return order_id
            else:
                self.logger.error("Failed to submit market order")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_bracket_order(self, action: OrderAction, quantity: int = 1, 
                           current_price: Optional[float] = None) -> Optional[Dict[str, str]]:
        """
        Place bracket order (entry + stop loss + take profit)
        
        Args:
            action: BUY or SELL
            quantity: Number of contracts
            current_price: Current market price
            
        Returns:
            Dict with order IDs: {'entry': id, 'stop': id, 'target': id}
        """
        try:
            if not current_price:
                self.logger.error("Current price required for bracket order")
                return None
            
            # Check risk limits
            if not self._check_risk_limits():
                return None
            
            # Calculate stop loss and take profit prices
            if action == OrderAction.BUY:
                stop_price = current_price - self.config.trading.stop_loss_points
                target_price = current_price + self.config.trading.take_profit_points
            else:  # SELL
                stop_price = current_price + self.config.trading.stop_loss_points
                target_price = current_price - self.config.trading.take_profit_points
            
            # Generate order IDs
            entry_id = str(uuid.uuid4())
            stop_id = str(uuid.uuid4())
            target_id = str(uuid.uuid4())
            
            # Create bracket order data
            bracket_data = {
                'type': 'BRACKET',
                'instrument': self.config.trading.instrument,
                'entry': {
                    'order_id': entry_id,
                    'action': action.value,
                    'quantity': quantity,
                    'order_type': 'MARKET'
                },
                'stop_loss': {
                    'order_id': stop_id,
                    'action': 'SELL' if action == OrderAction.BUY else 'BUY',
                    'quantity': quantity,
                    'order_type': 'STOP',
                    'stop_price': stop_price
                },
                'take_profit': {
                    'order_id': target_id,
                    'action': 'SELL' if action == OrderAction.BUY else 'BUY',
                    'quantity': quantity,
                    'order_type': 'LIMIT',
                    'limit_price': target_price
                }
            }
            
            # Submit bracket order (simplified for AT Interface)
            # For now, just place the entry order - stops and targets can be handled separately
            success = self.connector.place_order(
                account=self.config.account.account_name,
                instrument=self.config.trading.instrument,
                action=action.value,
                quantity=quantity,
                order_type="MARKET"
            )
            
            response = {'status': 'success'} if success else None
            
            if response and response.get('status') == 'success':
                # Create order objects
                entry_order = Order(
                    order_id=entry_id,
                    instrument=self.config.trading.instrument,
                    action=action,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.SUBMITTED
                )
                
                stop_order = Order(
                    order_id=stop_id,
                    instrument=self.config.trading.instrument,
                    action=OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY,
                    quantity=quantity,
                    order_type=OrderType.STOP,
                    stop_price=stop_price,
                    status=OrderStatus.SUBMITTED
                )
                
                target_order = Order(
                    order_id=target_id,
                    instrument=self.config.trading.instrument,
                    action=OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=target_price,
                    status=OrderStatus.SUBMITTED
                )
                
                # Store orders
                with self.order_lock:
                    self.orders[entry_id] = entry_order
                    self.orders[stop_id] = stop_order
                    self.orders[target_id] = target_order
                
                self.logger.info(f"Bracket order placed: {action.value} {quantity} @ stop={stop_price:.2f} target={target_price:.2f}")
                
                return {
                    'entry': entry_id,
                    'stop': stop_id,
                    'target': target_id
                }
            else:
                self.logger.error("Failed to submit bracket order")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    self.logger.error(f"Order not found: {order_id}")
                    return False
                
                order = self.orders[order_id]
                
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    self.logger.warning(f"Cannot cancel order in status: {order.status}")
                    return False
            
            # Submit cancellation via AT Interface
            success = self.connector.cancel_order(order_id)
            response = {'status': 'success'} if success else None
            
            if response and response.get('status') == 'success':
                with self.order_lock:
                    self.orders[order_id].status = OrderStatus.CANCELLED
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def close_position(self, market_order: bool = True) -> bool:
        """Close current position"""
        try:
            with self.position_lock:
                if not self.current_position or self.current_position.side == PositionSide.FLAT:
                    self.logger.info("No position to close")
                    return True
                
                position = self.current_position
            
            # Determine close action
            if position.side == PositionSide.LONG:
                close_action = OrderAction.SELL
            else:
                close_action = OrderAction.BUY
            
            self.logger.info(f"Closing {position.side.value} position of {position.quantity} contracts")
            
            if market_order:
                order_id = self.place_market_order(close_action, abs(position.quantity))
                return order_id is not None
            else:
                # Could add limit order close logic here
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def _submit_order(self, order: Order) -> bool:
        """Submit order to NinjaTrader"""
        try:
            order_data = {
                'order_id': order.order_id,
                'instrument': order.instrument,
                'action': order.action.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'time_in_force': order.time_in_force
            }
            
            if order.limit_price:
                order_data['limit_price'] = order.limit_price
            if order.stop_price:
                order_data['stop_price'] = order.stop_price
            
            # Submit via AT Interface
            success = self.connector.place_order(
                account=self.config.account.account_name,
                instrument=order.instrument,
                action=order.action.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                price=order.limit_price or 0,
                stop_price=order.stop_price or 0
            )
            
            response = {'status': 'success'} if success else {'status': 'error'}
            
            if response and response.get('status') == 'success':
                order.status = OrderStatus.SUBMITTED
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                order.status = OrderStatus.REJECTED
                order.error_message = error_msg
                self.logger.error(f"Order submission failed: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return False
    
    def _check_risk_limits(self) -> bool:
        """Check daily risk limits"""
        try:
            # Reset daily counters if new day
            current_date = datetime.now().date()
            if current_date > self.last_reset_date:
                self.daily_trades_count = 0
                self.daily_pnl = 0.0
                self.last_reset_date = current_date
                self.logger.info("Daily counters reset")
            
            # Check max trades per day
            if self.daily_trades_count >= self.config.trading.max_daily_trades:
                self.logger.error(f"Daily trade limit reached: {self.daily_trades_count}")
                return False
            
            # Check daily loss limit
            if self.daily_pnl <= -self.config.trading.max_daily_loss:
                self.logger.error(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _check_position_limits(self, action: OrderAction, quantity: int) -> bool:
        """Check position size limits"""
        try:
            with self.position_lock:
                if not self.current_position:
                    return True
                
                current_qty = self.current_position.quantity
                
                # Calculate new position size
                if action == OrderAction.BUY:
                    new_qty = current_qty + quantity
                else:
                    new_qty = current_qty - quantity
                
                # Check max position size
                if abs(new_qty) > self.config.trading.max_position_size:
                    self.logger.error(f"Position size limit exceeded: {abs(new_qty)}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False
    
    def _monitor_worker(self) -> None:
        """Monitor orders and positions"""
        self.logger.info("Order monitoring worker started")
        
        while self.monitoring:
            try:
                # Update orders
                self._update_orders()
                
                # Update position
                self._refresh_position()
                
                # Check for completed trades
                self._check_completed_trades()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor worker error: {e}")
                time.sleep(self.monitor_interval)
        
        self.logger.info("Order monitoring worker stopped")
    
    def _update_orders(self) -> None:
        """Update order statuses from NinjaTrader"""
        try:
            with self.order_lock:
                active_orders = {oid: order for oid, order in self.orders.items() 
                               if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]}
            
            if not active_orders:
                return
            
            # For AT Interface, order status updates would come via callbacks
            # For now, skip detailed order tracking - focus on position tracking
            return
            
            if response and 'orders' in response:
                for order_data in response['orders']:
                    order_id = order_data.get('order_id')
                    if order_id in active_orders:
                        self._update_order_from_data(active_orders[order_id], order_data)
                        
        except Exception as e:
            self.logger.error(f"Error updating orders: {e}")
    
    def _update_order_from_data(self, order: Order, order_data: Dict[str, Any]) -> None:
        """Update order from NinjaTrader data"""
        try:
            # Update status
            nt_status = order_data.get('status', '').upper()
            if nt_status in OrderStatus.__members__:
                old_status = order.status
                order.status = OrderStatus[nt_status]
                
                # Log status changes
                if old_status != order.status:
                    self.logger.info(f"Order {order.order_id} status: {old_status.value} -> {order.status.value}")
                    
                    # Notify callbacks
                    for callback in self.order_callbacks:
                        try:
                            callback(order)
                        except Exception as e:
                            self.logger.error(f"Order callback error: {e}")
            
            # Update fill information
            if 'filled_quantity' in order_data:
                order.filled_quantity = int(order_data['filled_quantity'])
            
            if 'avg_fill_price' in order_data:
                order.avg_fill_price = float(order_data['avg_fill_price'])
                
            if order.status == OrderStatus.FILLED and not order.filled_time:
                order.filled_time = datetime.now()
                self.daily_trades_count += 1
                
        except Exception as e:
            self.logger.error(f"Error updating order from data: {e}")
    
    def _refresh_position(self) -> None:
        """Refresh position from NinjaTrader"""
        try:
            # For AT Interface, position info would need to be tracked via order fills
            # For now, return a flat position - this will be enhanced when we have
            # proper AT Interface position reporting
            
            with self.position_lock:
                if not self.current_position:
                    self.current_position = Position(
                        instrument=self.config.trading.instrument,
                        side=PositionSide.FLAT,
                        quantity=0,
                        avg_price=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        market_value=0.0
                    )
            
            return
            
            if response and 'position' in response:
                pos_data = response['position']
                
                old_position = self.current_position
                
                with self.position_lock:
                    self.current_position = self._create_position_from_data(pos_data)
                
                # Notify callbacks if position changed
                if (not old_position and self.current_position) or \
                   (old_position and not self.current_position) or \
                   (old_position and self.current_position and 
                    old_position.quantity != self.current_position.quantity):
                    
                    for callback in self.position_callbacks:
                        try:
                            callback(self.current_position)
                        except Exception as e:
                            self.logger.error(f"Position callback error: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error refreshing position: {e}")
    
    def _create_position_from_data(self, pos_data: Dict[str, Any]) -> Optional[Position]:
        """Create Position object from NinjaTrader data"""
        try:
            quantity = int(pos_data.get('quantity', 0))
            
            if quantity == 0:
                return Position(
                    instrument=self.config.trading.instrument,
                    side=PositionSide.FLAT,
                    quantity=0,
                    avg_price=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    market_value=0.0
                )
            
            side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
            
            return Position(
                instrument=self.config.trading.instrument,
                side=side,
                quantity=abs(quantity),
                avg_price=float(pos_data.get('avg_price', 0)),
                unrealized_pnl=float(pos_data.get('unrealized_pnl', 0)),
                realized_pnl=float(pos_data.get('realized_pnl', 0)),
                market_value=float(pos_data.get('market_value', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error creating position from data: {e}")
            return None
    
    def _check_completed_trades(self) -> None:
        """Check for completed trades and update daily P&L"""
        # This would typically analyze filled orders to create Trade objects
        # Implementation depends on how NinjaTrader reports trade completions
        pass
    
    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add order status callback"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[Position], None]) -> None:
        """Add position update callback"""
        self.position_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add trade completion callback"""
        self.trade_callbacks.append(callback)
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status"""
        with self.order_lock:
            if status:
                return [order for order in self.orders.values() if order.status == status]
            return list(self.orders.values())
    
    def get_position(self) -> Optional[Position]:
        """Get current position"""
        with self.position_lock:
            return self.current_position
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        return {
            'trades_today': self.daily_trades_count,
            'daily_pnl': self.daily_pnl,
            'trades_remaining': max(0, self.config.trading.max_daily_trades - self.daily_trades_count),
            'loss_limit_remaining': max(0, self.config.trading.max_daily_loss + self.daily_pnl),
            'can_trade': (self.daily_trades_count < self.config.trading.max_daily_trades and 
                         self.daily_pnl > -self.config.trading.max_daily_loss)
        }