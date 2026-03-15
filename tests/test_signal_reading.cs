/*
Signal File Reading Test (C# Console Application)
Mimics the NinjaScript ESSignalExecutor signal reading logic
Compile with: csc test_signal_reading.cs
Run with: test_signal_reading.exe
*/

using System;
using System.IO;

class SignalReaderTest
{
    private static string signalFilePath = @"D:\QC_TradingBot_v3\data\bridge\signals.txt";
    
    static void Main(string[] args)
    {
        Console.WriteLine("NinjaScript Signal Reading Test");
        Console.WriteLine("=" + new string('=', 40));
        Console.WriteLine($"Testing signal file: {signalFilePath}");
        Console.WriteLine($"Timestamp: {DateTime.Now}");
        Console.WriteLine();
        
        // Test file existence
        TestFileExistence();
        
        // Test file reading
        TestFileReading();
        
        // Test signal parsing
        TestSignalParsing();
        
        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
    
    static void TestFileExistence()
    {
        Console.WriteLine("1. FILE EXISTENCE TEST");
        Console.WriteLine(new string('-', 25));
        
        if (!File.Exists(signalFilePath))
        {
            Console.WriteLine($"[ERROR] Signal file does not exist: {signalFilePath}");
            Console.WriteLine("        Run the Python trading bot first to create signals.");
            return;
        }
        
        Console.WriteLine($"[OK] Signal file exists: {signalFilePath}");
        
        // Check file properties
        FileInfo fileInfo = new FileInfo(signalFilePath);
        Console.WriteLine($"[INFO] File size: {fileInfo.Length} bytes");
        Console.WriteLine($"[INFO] Last modified: {fileInfo.LastWriteTime}");
        Console.WriteLine($"[INFO] File age: {(DateTime.Now - fileInfo.LastWriteTime).TotalSeconds:F1} seconds");
        Console.WriteLine();
    }
    
    static void TestFileReading()
    {
        Console.WriteLine("2. FILE READING TEST");
        Console.WriteLine(new string('-', 20));
        
        try
        {
            string signalData = File.ReadAllText(signalFilePath).Trim();
            
            if (string.IsNullOrEmpty(signalData))
            {
                Console.WriteLine("[WARNING] Signal file is empty");
                return;
            }
            
            Console.WriteLine($"[OK] Successfully read signal data");
            Console.WriteLine($"[INFO] Raw signal: '{signalData}'");
            Console.WriteLine($"[INFO] Length: {signalData.Length} characters");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to read signal file: {ex.Message}");
            Console.WriteLine();
            return;
        }
    }
    
    static void TestSignalParsing()
    {
        Console.WriteLine("3. SIGNAL PARSING TEST");
        Console.WriteLine(new string('-', 22));
        
        try
        {
            string signalData = File.ReadAllText(signalFilePath).Trim();
            
            if (string.IsNullOrEmpty(signalData))
            {
                Console.WriteLine("[WARNING] No signal data to parse");
                return;
            }
            
            // Parse signal exactly like NinjaScript does
            Console.WriteLine($"[INFO] Parsing signal: '{signalData}'");
            
            // Split by comma (same as NinjaScript)
            string[] parts = signalData.Split(',');
            Console.WriteLine($"[INFO] Split into {parts.Length} parts");
            
            for (int i = 0; i < parts.Length; i++)
            {
                Console.WriteLine($"       Part {i}: '{parts[i]}'");
            }
            
            if (parts.Length < 4)
            {
                Console.WriteLine($"[ERROR] Invalid signal format - expected 4 parts, got {parts.Length}");
                return;
            }
            
            // Parse each component (same validation as NinjaScript)
            DateTime signalTime;
            if (!DateTime.TryParse(parts[0], out signalTime))
            {
                Console.WriteLine($"[ERROR] Invalid timestamp format: '{parts[0]}'");
                return;
            }
            Console.WriteLine($"[OK] Parsed timestamp: {signalTime}");
            
            string action = parts[1].Trim().ToUpper();
            Console.WriteLine($"[OK] Parsed action: '{action}'");
            
            int quantity;
            if (!int.TryParse(parts[2], out quantity))
            {
                Console.WriteLine($"[ERROR] Invalid quantity: '{parts[2]}'");
                return;
            }
            Console.WriteLine($"[OK] Parsed quantity: {quantity}");
            
            double confidence;
            if (!double.TryParse(parts[3], out confidence))
            {
                Console.WriteLine($"[ERROR] Invalid confidence: '{parts[3]}'");
                return;
            }
            Console.WriteLine($"[OK] Parsed confidence: {confidence:F6} ({confidence:P2})");
            
            // Check signal age (same as NinjaScript)
            TimeSpan signalAge = DateTime.Now - signalTime;
            Console.WriteLine($"[INFO] Signal age: {signalAge.TotalSeconds:F1} seconds");
            
            if (signalAge.TotalSeconds > 10)
            {
                Console.WriteLine($"[WARNING] Signal is stale (> 10 seconds old)");
                Console.WriteLine($"          NinjaScript will reject this signal");
            }
            else
            {
                Console.WriteLine($"[OK] Signal is fresh (< 10 seconds old)");
            }
            
            // Check confidence threshold (same as NinjaScript)
            if (confidence < 0.3)
            {
                Console.WriteLine($"[WARNING] Confidence too low: {confidence:F4} < 0.3");
                Console.WriteLine($"          NinjaScript will reject this signal");
            }
            else
            {
                Console.WriteLine($"[OK] Confidence acceptable: {confidence:F4} >= 0.3");
            }
            
            // Validate action
            if (action != "BUY" && action != "SELL" && action != "FLAT" && action != "HOLD")
            {
                Console.WriteLine($"[WARNING] Unknown action: '{action}'");
                Console.WriteLine($"          Expected: BUY, SELL, FLAT, or HOLD");
            }
            else
            {
                Console.WriteLine($"[OK] Action is valid: '{action}'");
            }
            
            Console.WriteLine();
            Console.WriteLine("PARSING SUMMARY:");
            Console.WriteLine($"  Timestamp: {signalTime:yyyy-MM-dd HH:mm:ss.fff}");
            Console.WriteLine($"  Action: {action}");
            Console.WriteLine($"  Quantity: {quantity}");
            Console.WriteLine($"  Confidence: {confidence:P2}");
            Console.WriteLine($"  Age: {signalAge.TotalSeconds:F1}s");
            Console.WriteLine($"  Valid for NinjaScript: {(signalAge.TotalSeconds <= 10 && confidence >= 0.3 ? "YES" : "NO")}");
            
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Parsing failed: {ex.Message}");
            Console.WriteLine($"[DEBUG] Stack trace: {ex.StackTrace}");
        }
    }
}