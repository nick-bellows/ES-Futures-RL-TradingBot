using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ESSignalExecutor : Strategy
    {
        private string signalFilePath = @"D:\QC_TradingBot_v3\data\bridge\signals.txt";
        private string logFilePath = @"D:\QC_TradingBot_v3\data\bridge\execution_log.txt";
        private DateTime lastSignalCheck = DateTime.MinValue;
        private string lastProcessedSignal = "";
        private object fileLock = new object();
        
        // Trading parameters
        private int maxPosition = 1;
        private double stopLossPoints = 5.0;
        private double profitTargetPoints = 15.0;
        private double minPriceChange = 0.25; // Minimum tick size
        
        // Position tracking
        private DateTime lastOrderTime = DateTime.MinValue;
        private DateTime lastPositionCloseTime = DateTime.MinValue;
        private MarketPosition lastClosedPosition = MarketPosition.Flat;
        private double currentStopPrice = 0;
        private double currentTargetPrice = 0;
        private double entryPrice = 0;
        private bool hasActiveOrders = false;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"ES Signal Executor - Executes Python ML bot signals with risk management";
                Name = "ESSignalExecutor";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = true;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 1;
                IsInstantiatedOnEachOptimizationIteration = true;
                
                // Initialize parameters
                StopLossPoints = 5.0;
                ProfitTargetPoints = 15.0;
                MaxPositionSize = 1;
                SignalCheckInterval = 1; // Check every 1 second
            }
            else if (State == State.Active)
            {
                InitializeSignalExecutor();
            }
        }

        protected override void OnBarUpdate()
        {
            // Check for new signals every bar close or every N seconds
            DateTime now = DateTime.Now;
            if (now - lastSignalCheck >= TimeSpan.FromSeconds(SignalCheckInterval))
            {
                CheckForNewSignals();
                lastSignalCheck = now;
            }
            
            // Update stop loss if in position
            if (Position.MarketPosition == MarketPosition.Long && Position.Quantity > 0)
            {
                UpdateTrailingStop();
            }
        }

        private void InitializeSignalExecutor()
        {
            try
            {
                // Create directory if needed
                string directory = Path.GetDirectoryName(signalFilePath);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // Clear signal file
                lock (fileLock)
                {
                    if (File.Exists(signalFilePath))
                    {
                        File.Delete(signalFilePath);
                    }
                }

                LogExecution("INITIALIZED", "Signal executor started");
                Print("ESSignalExecutor initialized - monitoring " + signalFilePath);
            }
            catch (Exception ex)
            {
                Print($"ERROR initializing signal executor: {ex.Message}");
            }
        }

        private void CheckForNewSignals()
        {
            try
            {
                if (!File.Exists(signalFilePath))
                    return;

                string signalData = "";
                lock (fileLock)
                {
                    // Read signal file
                    signalData = File.ReadAllText(signalFilePath).Trim();
                }

                if (string.IsNullOrEmpty(signalData) || signalData == lastProcessedSignal)
                    return;

                // Parse signal: timestamp,action,quantity,confidence
                string[] parts = signalData.Split(',');
                if (parts.Length < 4)
                {
                    LogExecution("ERROR", $"Invalid signal format: {signalData}");
                    return;
                }

                DateTime signalTime = DateTime.Parse(parts[0]);
                string action = parts[1].Trim().ToUpper();
                int quantity = int.Parse(parts[2]);
                double confidence = double.Parse(parts[3]);

                // Check signal freshness (must be < 10 seconds old)
                TimeSpan signalAge = DateTime.Now - signalTime;
                if (signalAge.TotalSeconds > 10)
                {
                    LogExecution("STALE", $"Signal too old: {signalAge.TotalSeconds:F1}s");
                    return;
                }

                // Check minimum confidence
                if (confidence < 0.45)
                {
                    LogExecution("LOW_CONFIDENCE", $"Signal confidence too low: {confidence:F4} (min: 0.45)");
                    return;
                }

                // Process the signal
                ProcessSignal(action, quantity, confidence, signalTime);
                lastProcessedSignal = signalData;

            }
            catch (Exception ex)
            {
                LogExecution("ERROR", $"Error checking signals: {ex.Message}");
                Print($"ERROR checking signals: {ex.Message}");
            }
        }

        private void ProcessSignal(string action, int quantity, double confidence, DateTime signalTime)
        {
            try
            {
                // Validate quantity
                if (quantity > MaxPositionSize)
                {
                    quantity = MaxPositionSize;
                    LogExecution("QUANTITY_LIMITED", $"Reduced quantity to {quantity}");
                }

                // Check current position
                MarketPosition currentPosition = Position.MarketPosition;
                int currentQuantity = Position.Quantity;

                LogExecution("SIGNAL_RECEIVED", $"{action} {quantity} (conf: {confidence:F4}, pos: {currentPosition} {currentQuantity})");

                // SAFETY CHECK 1: Position flip protection (60-second cooldown)
                DateTime now = DateTime.Now;
                if (lastPositionCloseTime != DateTime.MinValue)
                {
                    TimeSpan timeSinceClose = now - lastPositionCloseTime;
                    if (timeSinceClose.TotalSeconds < 60)
                    {
                        LogExecution("COOLDOWN", $"Position flip blocked - {timeSinceClose.TotalSeconds:F1}s since last close (need 60s)");
                        return;
                    }
                }

                // SAFETY CHECK 2: Prevent rapid position flips
                bool wouldFlipPosition = false;
                if ((action == "BUY" && lastClosedPosition == MarketPosition.Short) ||
                    (action == "SELL" && lastClosedPosition == MarketPosition.Long))
                {
                    TimeSpan timeSinceClose = now - lastPositionCloseTime;
                    if (timeSinceClose.TotalSeconds < 120) // 2-minute minimum between flips
                    {
                        LogExecution("FLIP_BLOCKED", $"Position flip blocked - {timeSinceClose.TotalSeconds:F1}s since {lastClosedPosition} close");
                        return;
                    }
                }

                // SAFETY CHECK 3: Ignore duplicate direction signals
                if ((action == "BUY" && currentPosition == MarketPosition.Long) ||
                    (action == "SELL" && currentPosition == MarketPosition.Short))
                {
                    LogExecution("IGNORED", $"Already {action.ToLower()}");
                    return;
                }

                switch (action)
                {
                    case "BUY":
                        if (currentPosition == MarketPosition.Flat)
                        {
                            ExecuteBuyOrder(quantity, confidence);
                        }
                        else if (currentPosition == MarketPosition.Short)
                        {
                            // Close short first, then go long (with safety checks passed)
                            ExitShort();
                            lastPositionCloseTime = DateTime.Now;
                            lastClosedPosition = MarketPosition.Short;
                            LogExecution("POSITION_CLOSED", "Short closed for BUY signal");
                            // Don't immediately enter long - wait for next signal
                            return;
                        }
                        break;

                    case "SELL":
                        if (currentPosition == MarketPosition.Flat)
                        {
                            ExecuteSellOrder(quantity, confidence);
                        }
                        else if (currentPosition == MarketPosition.Long)
                        {
                            // Close long first, then go short (with safety checks passed)
                            ExitLong();
                            lastPositionCloseTime = DateTime.Now;
                            lastClosedPosition = MarketPosition.Long;
                            LogExecution("POSITION_CLOSED", "Long closed for SELL signal");
                            // Don't immediately enter short - wait for next signal
                            return;
                        }
                        break;

                    case "FLAT":
                    case "CLOSE":
                        if (currentPosition != MarketPosition.Flat)
                        {
                            CloseAllPositions();
                            lastPositionCloseTime = DateTime.Now;
                            lastClosedPosition = currentPosition;
                        }
                        else
                        {
                            LogExecution("IGNORED", "Already flat");
                        }
                        break;

                    default:
                        LogExecution("ERROR", $"Unknown action: {action}");
                        break;
                }

            }
            catch (Exception ex)
            {
                LogExecution("ERROR", $"Error processing signal: {ex.Message}");
            }
        }

        private void ExecuteBuyOrder(int quantity, double confidence)
        {
            try
            {
                // Prevent rapid orders
                if (DateTime.Now - lastOrderTime < TimeSpan.FromSeconds(5))
                {
                    LogExecution("THROTTLED", "Order too soon after last order");
                    return;
                }

                // Enter long position
                EnterLong(quantity, "MLBuy");
                lastOrderTime = DateTime.Now;
                
                // Calculate stop and target prices (ES: 1 point = 1.0 price, not TickSize)
                entryPrice = GetCurrentAsk();  // Store entry price for trailing stop
                currentStopPrice = entryPrice - StopLossPoints;  // Subtract 5 points directly
                currentTargetPrice = entryPrice + ProfitTargetPoints;  // Add 15 points directly

                LogExecution("BUY_ORDER", $"Qty: {quantity}, Entry: ~{entryPrice:F2}, Stop: {currentStopPrice:F2}, Target: {currentTargetPrice:F2}");

            }
            catch (Exception ex)
            {
                LogExecution("ERROR", $"Error executing buy order: {ex.Message}");
            }
        }

        private void ExecuteSellOrder(int quantity, double confidence)
        {
            try
            {
                // Prevent rapid orders
                if (DateTime.Now - lastOrderTime < TimeSpan.FromSeconds(5))
                {
                    LogExecution("THROTTLED", "Order too soon after last order");
                    return;
                }

                // Enter short position
                EnterShort(quantity, "MLSell");
                lastOrderTime = DateTime.Now;

                // Calculate stop and target prices (ES: 1 point = 1.0 price, not TickSize)
                entryPrice = GetCurrentBid();  // Store entry price for trailing stop
                currentStopPrice = entryPrice + StopLossPoints;  // Add 5 points directly (stop above for short)
                currentTargetPrice = entryPrice - ProfitTargetPoints;  // Subtract 15 points directly (target below for short)

                LogExecution("SELL_ORDER", $"Qty: {quantity}, Entry: ~{entryPrice:F2}, Stop: {currentStopPrice:F2}, Target: {currentTargetPrice:F2}");

            }
            catch (Exception ex)
            {
                LogExecution("ERROR", $"Error executing sell order: {ex.Message}");
            }
        }

        private void CloseAllPositions()
        {
            try
            {
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    ExitLong();
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    ExitShort();
                }

                LogExecution("CLOSE_ALL", "Closing all positions");
            }
            catch (Exception ex)
            {
                LogExecution("ERROR", $"Error closing positions: {ex.Message}");
            }
        }

        private void UpdateTrailingStop()
        {
            if (Position.MarketPosition == MarketPosition.Long && Position.Quantity > 0 && entryPrice > 0)
            {
                double currentPrice = Close[0];
                double profitPoints = currentPrice - entryPrice;

                // Only start trailing after we have at least 2.5 points profit (half of our stop distance)
                if (profitPoints < 2.5)
                {
                    return; // Not enough profit to start trailing
                }

                // Calculate new stop price: maintain full 5-point buffer from current price
                double newStopPrice = currentPrice - StopLossPoints;

                // Only move stop up, never down, and only if it improves our position
                if (newStopPrice > currentStopPrice)
                {
                    double oldStop = currentStopPrice;
                    currentStopPrice = newStopPrice;

                    // Update stop order
                    SetStopLoss("MLBuy", CalculationMode.Price, currentStopPrice, false);
                    LogExecution("TRAILING_STOP", $"Updated stop to {currentStopPrice:F2} (was {oldStop:F2}), profit: {profitPoints:F2} pts");
                }
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string nativeError)
        {
            // Log order updates
            if (order.Name == "MLBuy" || order.Name == "MLSell")
            {
                LogExecution("ORDER_UPDATE", $"{order.Name}: {orderState}, Filled: {filled}/{quantity} @ {averageFillPrice:F2}");
                
                if (orderState == OrderState.Filled)
                {
                    // Set stop loss and profit target
                    // ES Futures: 1 point = 4 ticks = $50
                    int stopTicks = (int)(StopLossPoints * 4);  // Convert points to ticks (1 point = 4 ticks)
                    int targetTicks = (int)(ProfitTargetPoints * 4);  // Convert points to ticks

                    if (order.Name == "MLBuy")
                    {
                        SetStopLoss("MLBuy", CalculationMode.Ticks, stopTicks);
                        SetProfitTarget("MLBuy", CalculationMode.Ticks, targetTicks);
                        LogExecution("STOP_TARGET_SET", $"Long: Stop={stopTicks} ticks ({StopLossPoints} pts), Target={targetTicks} ticks ({ProfitTargetPoints} pts)");
                    }
                    else if (order.Name == "MLSell")
                    {
                        SetStopLoss("MLSell", CalculationMode.Ticks, stopTicks);
                        SetProfitTarget("MLSell", CalculationMode.Ticks, targetTicks);
                        LogExecution("STOP_TARGET_SET", $"Short: Stop={stopTicks} ticks ({StopLossPoints} pts), Target={targetTicks} ticks ({ProfitTargetPoints} pts)");
                    }
                }
            }
        }

        private void LogExecution(string action, string message)
        {
            try
            {
                lock (fileLock)
                {
                    string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff},{action},{message}";
                    File.AppendAllText(logFilePath, logEntry + "\n");
                }
                
                // Also print to NinjaTrader output window
                Print($"[ESSignalExecutor] {action}: {message}");
                
            }
            catch (Exception ex)
            {
                Print($"ERROR writing log: {ex.Message}");
            }
        }

        #region Properties

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Stop Loss Points", Description = "Stop loss in points", Order = 1, GroupName = "Risk Management")]
        public double StopLossPoints { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Profit Target Points", Description = "Profit target in points", Order = 2, GroupName = "Risk Management")]
        public double ProfitTargetPoints { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Max Position Size", Description = "Maximum position size", Order = 3, GroupName = "Risk Management")]
        public int MaxPositionSize { get; set; }

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Signal Check Interval", Description = "Signal check interval in seconds", Order = 4, GroupName = "Timing")]
        public int SignalCheckInterval { get; set; }

        #endregion
    }
}

// Usage Instructions:
// 1. Copy this file to: Documents/NinjaTrader 8/bin/Custom/Strategies/
// 2. Compile in NinjaScript Editor
// 3. Apply strategy to your ES chart
// 4. Check C:\NTBridge\execution_log.txt for activity
// 5. Ensure C:\NTBridge\signals.txt exists (created by Python bot)