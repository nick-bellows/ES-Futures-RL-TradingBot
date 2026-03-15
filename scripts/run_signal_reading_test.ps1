# PowerShell script to compile and run the C# signal reading test
# Usage: .\run_signal_reading_test.ps1

Write-Host "NinjaScript Signal Reading Test" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if .NET compiler is available
$cscPath = Get-Command csc.exe -ErrorAction SilentlyContinue

if (-not $cscPath) {
    # Try to find csc.exe in common .NET Framework locations
    $dotnetPaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\Roslyn\csc.exe", 
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:WINDIR}\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
        "${env:WINDIR}\Microsoft.NET\Framework\v4.0.30319\csc.exe"
    )
    
    foreach ($path in $dotnetPaths) {
        if (Test-Path $path) {
            $cscPath = Get-Command $path
            break
        }
    }
}

if (-not $cscPath) {
    Write-Host "[ERROR] C# compiler (csc.exe) not found!" -ForegroundColor Red
    Write-Host "Please install .NET Framework Developer Pack or Visual Studio" -ForegroundColor Yellow
    Write-Host "Or run the test manually with: csc test_signal_reading.cs" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Using C# compiler: $($cscPath.Source)" -ForegroundColor Green

# Compile the C# test
Write-Host "[INFO] Compiling test_signal_reading.cs..." -ForegroundColor Yellow

try {
    & $cscPath.Source "test_signal_reading.cs" 2>&1 | Out-String | Write-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[OK] Compilation successful" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Compilation error: $_" -ForegroundColor Red
    exit 1
}

# Check if executable was created
if (-not (Test-Path "test_signal_reading.exe")) {
    Write-Host "[ERROR] Executable not found after compilation" -ForegroundColor Red
    exit 1
}

# Run the test
Write-Host "[INFO] Running signal reading test..." -ForegroundColor Yellow
Write-Host ""

try {
    & ".\test_signal_reading.exe"
    Write-Host ""
    Write-Host "[INFO] Test completed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Test execution failed: $_" -ForegroundColor Red
    exit 1
} finally {
    # Cleanup
    if (Test-Path "test_signal_reading.exe") {
        Remove-Item "test_signal_reading.exe" -Force
        Write-Host "[INFO] Cleaned up executable" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "If this test passes but NinjaTrader still doesn't process signals:" -ForegroundColor Cyan
Write-Host "1. Verify ESSignalExecutor.cs is installed in NinjaTrader" -ForegroundColor White
Write-Host "2. Check that the strategy is applied and ENABLED on ES chart" -ForegroundColor White
Write-Host "3. Monitor NinjaTrader Output window for debug messages" -ForegroundColor White
Write-Host "4. Run: python verify_ninjatrader_integration.py" -ForegroundColor White