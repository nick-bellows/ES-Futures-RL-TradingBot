using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;

namespace NinjaTrader.NinjaScript.Indicators
{
    public class ESDataBridge : Indicator
    {
        private string dataFilePath = @"C:\NTBridge\market_data.csv";
        private string statusFilePath = @"C:\NTBridge\status.txt";
        private DateTime lastWriteTime = DateTime.MinValue;
        private bool isSessionActive = false;
        private int tickCount = 0;
        private double sessionOpen = 0;
        private double sessionHigh = 0;
        private double sessionLow = 0;
        private object fileLock = new object();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"ES Data Bridge - Captures real-time market data for Python ML bot";
                Name = "ESDataBridge";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                DrawHorizontalGridLines = true;
                DrawVerticalGridLines = true;
                PaintPriceMarkers = true;
                ScaleJustification = ScaleJustification.Right;
                IsSuspendedWhileInactive = false;
                
                // Visual indicators
                ShowStatusText = true;
                StatusBackgroundColor = Brushes.DarkBlue;
                StatusTextColor = Brushes.White;
            }
            else if (State == State.Active)
            {
                InitializeDataBridge();
            }
            else if (State == State.Terminated)
            {
                WriteStatus("TERMINATED", "Data bridge stopped");
            }
        }

        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            if (State != State.Active) return;

            try
            {
                // Get current market data
                double bid = GetCurrentBid();
                double ask = GetCurrentAsk();
                double last = GetCurrentAsk() != 0 ? (bid + ask) / 2 : Close[0]; // Use mid-price or close
                long volume = Volume[0];
                
                // Track session data
                if (!isSessionActive)
                {
                    sessionOpen = last;
                    sessionHigh = last;
                    sessionLow = last;
                    isSessionActive = true;
                    Print($"Session started at {last}");
                }
                else
                {
                    sessionHigh = Math.Max(sessionHigh, last);
                    sessionLow = Math.Min(sessionLow, last);
                }

                // Write market data (every tick)
                WriteMarketData(bid, ask, last, volume);
                
                tickCount++;
                
                // Update status every 100 ticks
                if (tickCount % 100 == 0)
                {
                    WriteStatus("ACTIVE", $"Ticks: {tickCount}, Last: {last:F2}");
                }

            }
            catch (Exception ex)
            {
                Print($"ERROR in OnMarketData: {ex.Message}");
                WriteStatus("ERROR", ex.Message);
            }
        }

        protected override void OnBarUpdate()
        {
            // Update bar-based data
            if (BarsInProgress == 0)
            {
                // Ensure we have session data
                if (!isSessionActive)
                {
                    sessionOpen = Open[0];
                    sessionHigh = High[0];
                    sessionLow = Low[0];
                    isSessionActive = true;
                }
            }
        }

        private void InitializeDataBridge()
        {
            try
            {
                // Create directory if it doesn't exist
                string directory = Path.GetDirectoryName(dataFilePath);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                    Print($"Created directory: {directory}");
                }

                // Clear/initialize data file with header
                lock (fileLock)
                {
                    File.WriteAllText(dataFilePath, "timestamp,bid,ask,last,volume,high,low,open,close\n");
                }

                tickCount = 0;
                Print($"ESDataBridge initialized - writing to {dataFilePath}");
                WriteStatus("INITIALIZED", "Data bridge started");
            }
            catch (Exception ex)
            {
                Print($"ERROR initializing data bridge: {ex.Message}");
            }
        }

        private void WriteMarketData(double bid, double ask, double last, long volume)
        {
            try
            {
                // Throttle writes - only write if price changed or 1 second passed
                DateTime now = DateTime.Now;
                if (now - lastWriteTime < TimeSpan.FromMilliseconds(250)) // Max 4 times per second
                    return;

                lastWriteTime = now;

                // Format: timestamp,bid,ask,last,volume,high,low,open,close
                string dataLine = string.Format("{0:yyyy-MM-dd HH:mm:ss.fff},{1:F2},{2:F2},{3:F2},{4},{5:F2},{6:F2},{7:F2},{8:F2}",
                    now, bid, ask, last, volume, sessionHigh, sessionLow, sessionOpen, Close[0]);

                // Atomic write with file locking
                lock (fileLock)
                {
                    // Write to temporary file first, then rename (atomic operation)
                    string tempFile = dataFilePath + ".tmp";
                    File.WriteAllText(tempFile, dataLine + "\n");
                    File.Move(tempFile, dataFilePath);
                }

            }
            catch (Exception ex)
            {
                Print($"ERROR writing market data: {ex.Message}");
            }
        }

        private void WriteStatus(string status, string message)
        {
            try
            {
                lock (fileLock)
                {
                    string statusData = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff},{status},{message}";
                    File.WriteAllText(statusFilePath, statusData);
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR writing status: {ex.Message}");
            }
        }

        public override void OnRenderTargetChanged()
        {
            // Visual status indicator on chart
            if (ShowStatusText && ChartControl != null)
            {
                // This will show status on chart
            }
        }

        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            if (ShowStatusText)
            {
                // Draw status text on chart
                string statusText = $"ES Data Bridge: {tickCount} ticks";
                
                // Position in top-left corner
                var textBrush = StatusTextColor;
                var backgroundBrush = StatusBackgroundColor;
                
                // Simple text rendering
                System.Windows.Point textPoint = new System.Windows.Point(10, 30);
                
                // Draw background rectangle
                System.Windows.Rect backgroundRect = new System.Windows.Rect(5, 25, 200, 25);
                RenderTarget.FillRectangle(backgroundRect, backgroundBrush);
                
                // Draw text
                var format = new System.Windows.Media.FormattedText(statusText,
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Windows.FlowDirection.LeftToRight,
                    new System.Windows.Media.Typeface("Arial"),
                    12, textBrush, 96);
                
                RenderTarget.DrawText(format, textPoint);
            }
        }

        #region Properties

        [NinjaScriptProperty]
        [Display(Name = "Show Status Text", Description = "Show status text on chart", Order = 1, GroupName = "Display")]
        public bool ShowStatusText { get; set; }

        [XmlIgnore]
        [Display(Name = "Status Background", Description = "Background color for status text", Order = 2, GroupName = "Display")]
        public Brush StatusBackgroundColor { get; set; }

        [XmlIgnore]
        [Display(Name = "Status Text Color", Description = "Text color for status", Order = 3, GroupName = "Display")]
        public Brush StatusTextColor { get; set; }

        #endregion
    }
}

// Usage Instructions:
// 1. Copy this file to: Documents/NinjaTrader 8/bin/Custom/Indicators/
// 2. Compile in NinjaScript Editor
// 3. Add indicator to your ES chart
// 4. Check that C:\NTBridge\market_data.csv is being updated
// 5. Look for status text in top-left of chart