"""
QuantConnect ES Futures PPO Trading Algorithm (Python)
Implements the trained PPO model for live trading
"""

from AlgorithmImports import *
from ppo_inference import PPOInference, FeatureCalculator


class ESFuturesPPOAlgorithm(QCAlgorithm):
    """QuantConnect algorithm using trained PPO model for ES Futures trading"""
    
    def Initialize(self):
        """Initialize the algorithm"""
        
        # Algorithm settings
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Add ES Futures data
        self.es_future = self.AddFuture(Futures.Indices.SP500EMini)
        self.es_future.SetFilter(0, 182)  # 0 to 6 months expiry
        
        # Initialize PPO model
        self.ppo_model = PPOInference("ppo_weights.json")
        if not self.ppo_model.load_weights():
            raise Exception("Failed to load PPO model weights")
        
        # Initialize feature calculator
        self.features = FeatureCalculator(lookback_periods=60)
        
        # Trading state
        self.current_contract = None
        self.last_price = 0
        self.position_value = 0
        
        # Risk management
        self.max_daily_loss = -1500  # Stop trading if daily loss exceeds $1500
        self.profit_target = 750     # 15 points * $50 per point
        self.stop_loss = -250        # 5 points * $50 per point
        self.daily_pnl = 0
        self.trades_today = 0
        self.max_trades_per_day = 10
        
        # Performance tracking
        self.model_predictions = 0
        self.action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        # Schedule daily reset
        self.Schedule.On(
            self.DateRules.EveryDay("ES"),
            self.TimeRules.At(16, 0),  # 4 PM EST market close
            self.DailyReset
        )
        
        # Schedule trading (9:30 AM to 4 PM EST)
        self.Schedule.On(
            self.DateRules.EveryDay("ES"),
            self.TimeRules.Every(TimeSpan.FromMinutes(1)),
            self.ExecuteTrading
        )
        
        self.Log("PPO Algorithm Initialized - Using STOCHASTIC evaluation for 36.5% target win rate")
    
    def OnData(self, data):
        """Process incoming data"""
        
        # Get current ES contract
        for chain in data.FutureChains:
            if chain.Key != self.es_future.Symbol:
                continue
                
            # Select the front month contract
            contracts = [x for x in chain.Value if x.Expiry > self.Time]
            if not contracts:
                continue
                
            # Sort by expiry and take the nearest
            contracts = sorted(contracts, key=lambda x: x.Expiry)
            self.current_contract = contracts[0]
            
            # Update last price
            if self.current_contract.BidPrice > 0 and self.current_contract.AskPrice > 0:
                self.last_price = (self.current_contract.BidPrice + self.current_contract.AskPrice) / 2
            elif self.current_contract.LastPrice > 0:
                self.last_price = self.current_contract.LastPrice
            
            break
    
    def ExecuteTrading(self):
        """Main trading logic called every minute"""
        
        if not self.current_contract or self.last_price <= 0:
            return
        
        # Risk management checks
        if self.daily_pnl <= self.max_daily_loss:
            self.Log(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return
        
        if self.trades_today >= self.max_trades_per_day:
            self.Log(f"Max trades per day reached: {self.trades_today}")
            return
        
        # Update feature calculator with current market data
        current_volume = self.current_contract.Volume if hasattr(self.current_contract, 'Volume') else 0
        features_vector = self.features.calculate_features(self.last_price, current_volume)
        
        # Get model prediction (CRITICAL: Use stochastic=True)
        action, confidence = self.ppo_model.predict(features_vector, stochastic=True)
        
        # Update statistics
        self.model_predictions += 1
        action_names = ['HOLD', 'BUY', 'SELL']
        self.action_counts[action_names[action]] += 1
        
        # Check current position
        current_position = self.Portfolio[self.current_contract.Symbol].Quantity
        
        # Position and risk management
        current_unrealized = self.Portfolio[self.current_contract.Symbol].UnrealizedProfitLoss
        
        # Stop loss and profit target
        if current_position != 0:
            if current_unrealized <= self.stop_loss:
                self.Log(f"Stop loss hit: ${current_unrealized:.2f}")
                self.Liquidate(self.current_contract.Symbol)
                self.trades_today += 1
                self.daily_pnl += current_unrealized
                self.features.update_position(0, self.last_price)  # Flat position
                return
            elif current_unrealized >= self.profit_target:
                self.Log(f"Profit target hit: ${current_unrealized:.2f}")
                self.Liquidate(self.current_contract.Symbol)
                self.trades_today += 1
                self.daily_pnl += current_unrealized
                self.features.update_position(0, self.last_price)  # Flat position
                return
        
        # Execute model action
        if action == 1 and current_position <= 0:  # BUY signal
            if current_position < 0:
                self.Log(f"Covering short position. Confidence: {confidence:.3f}")
            else:
                self.Log(f"Opening long position. Confidence: {confidence:.3f}")
            
            self.SetHoldings(self.current_contract.Symbol, 1.0)
            self.trades_today += 1
            self.features.update_position(action, self.last_price)
            
        elif action == 2 and current_position >= 0:  # SELL signal
            if current_position > 0:
                self.Log(f"Closing long position. Confidence: {confidence:.3f}")
            else:
                self.Log(f"Opening short position. Confidence: {confidence:.3f}")
            
            self.SetHoldings(self.current_contract.Symbol, -1.0)
            self.trades_today += 1
            self.features.update_position(action, self.last_price)
        
        # Log model state periodically
        if self.model_predictions % 60 == 0:  # Every hour
            stats = self.ppo_model.get_statistics()
            self.Log(f"Model Stats - Predictions: {stats['total_predictions']}, "
                    f"Actions: H{stats['action_distribution']['HOLD']:.1%} "
                    f"B{stats['action_distribution']['BUY']:.1%} "
                    f"S{stats['action_distribution']['SELL']:.1%}")
    
    def DailyReset(self):
        """Reset daily tracking variables"""
        
        # Calculate daily P&L
        total_pnl = sum([holding.UnrealizedProfitLoss + holding.LastTradeProfit 
                        for holding in self.Portfolio.Values])
        
        self.Log(f"Daily Summary - P&L: ${total_pnl:.2f}, Trades: {self.trades_today}")
        
        # Log model performance
        if self.model_predictions > 0:
            action_dist = {}
            for action, count in self.action_counts.items():
                action_dist[action] = count / self.model_predictions * 100
            
            self.Log(f"Model Action Distribution - HOLD: {action_dist['HOLD']:.1f}% "
                    f"BUY: {action_dist['BUY']:.1f}% SELL: {action_dist['SELL']:.1f}%")
        
        # Reset daily counters
        self.trades_today = 0
        self.daily_pnl = 0
        self.features.trades_today = 0
        self.features.daily_pnl = 0
    
    def OnEndOfDay(self):
        """End of day processing"""
        
        # Close all positions at end of day (no overnight positions)
        if self.Portfolio.Invested:
            self.Log("Closing positions - No overnight holds")
            self.Liquidate()
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        
        if orderEvent.Status == OrderStatus.Filled:
            order = self.Transactions.GetOrderById(orderEvent.OrderId)
            self.Log(f"Order Filled: {order.Symbol} {order.Quantity} @ ${orderEvent.FillPrice}")
            
            # Update daily P&L tracking
            if orderEvent.Direction == OrderDirection.Sell:
                self.daily_pnl += orderEvent.FillQuantity * orderEvent.FillPrice * -1
            else:
                self.daily_pnl += orderEvent.FillQuantity * orderEvent.FillPrice
    
    def OnSecuritiesChanged(self, changes):
        """Handle security additions/removals"""
        
        for security in changes.AddedSecurities:
            self.Log(f"Added security: {security.Symbol}")
            
        for security in changes.RemovedSecurities:
            self.Log(f"Removed security: {security.Symbol}")


# Additional utility functions for QuantConnect deployment
class ModelValidator:
    """Validate model performance in backtesting"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.trades = []
        self.daily_returns = []
        
    def record_trade(self, entry_price, exit_price, direction, duration):
        """Record trade for analysis"""
        pnl = (exit_price - entry_price) * direction * 50  # $50 per point
        
        self.trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,  # 1 for long, -1 for short
            'pnl': pnl,
            'duration': duration
        })
    
    def calculate_win_rate(self):
        """Calculate win rate"""
        if not self.trades:
            return 0.0
        
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        return winning_trades / len(self.trades)
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        pnls = [t['pnl'] for t in self.trades]
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'win_rate': self.calculate_win_rate(),
            'total_pnl': sum(pnls),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum([t['pnl'] for t in winning_trades]) / 
                                sum([t['pnl'] for t in losing_trades])) if losing_trades else float('inf'),
            'max_drawdown': min(pnls) if pnls else 0
        }
        
        return metrics


# Configuration for QuantConnect deployment
ALGORITHM_CONFIG = {
    "model_file": "ppo_weights.json",
    "use_stochastic": True,  # CRITICAL: Must be True for 36.5% win rate
    "risk_management": {
        "max_daily_loss": -1500,
        "profit_target_points": 15,
        "stop_loss_points": 5,
        "max_trades_per_day": 10
    },
    "trading_hours": {
        "start": "09:30",
        "end": "16:00",
        "timezone": "Eastern"
    },
    "expected_performance": {
        "target_win_rate": 0.365,  # 36.5% from validation testing
        "min_win_rate": 0.25,      # Break-even requirement
        "expected_trades_per_day": 3,
        "risk_reward_ratio": 3.0   # 15pt target / 5pt stop
    }
}