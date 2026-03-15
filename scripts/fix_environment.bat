@echo off
echo =========================================================
echo ES FUTURES RL TRADING BOT - ENVIRONMENT CLEAN INSTALL
echo =========================================================
echo.
echo This will completely reset your Python environment.
echo Press Ctrl+C to cancel, or any key to continue...
pause > nul

echo.
echo [1/8] Deactivating any active virtual environment...
if defined VIRTUAL_ENV (
    echo Deactivating: %VIRTUAL_ENV%
    deactivate 2>nul
) else (
    echo No virtual environment active
)

echo.
echo [2/8] Removing existing virtual environment folder...
if exist "venv" (
    echo Deleting venv folder...
    rmdir /s /q venv
    echo Venv folder deleted
) else (
    echo No existing venv folder found
)

echo.
echo [3/8] Creating fresh Python 3.11 virtual environment...
py -3.11 -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.11 is installed: py -3.11 --version
    pause
    exit /b 1
)
echo Virtual environment created successfully

echo.
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated

echo.
echo [5/8] Upgrading pip to latest version...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo.
echo [6/8] Installing core packages in correct dependency order...
echo.

echo Installing numpy (foundation package)...
python -m pip install numpy==1.26.4
if errorlevel 1 (
    echo ERROR: Failed to install numpy
    pause
    exit /b 1
)

echo Installing pandas (depends on numpy)...
python -m pip install pandas==2.1.4
if errorlevel 1 (
    echo ERROR: Failed to install pandas
    pause
    exit /b 1
)

echo Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA
    echo Trying CPU version as fallback...
    python -m pip install torch torchvision torchaudio
)

echo Installing scientific computing packages...
python -m pip install scipy scikit-learn matplotlib seaborn
if errorlevel 1 (
    echo WARNING: Some scientific packages failed to install
)

echo Installing RL and ML packages...
python -m pip install stable-baselines3 gymnasium
if errorlevel 1 (
    echo ERROR: Failed to install RL packages
    pause
    exit /b 1
)

echo Installing utilities and dependencies...
python -m pip install tensorboard pyyaml python-dotenv tqdm
if errorlevel 1 (
    echo WARNING: Some utility packages failed to install
)

echo Installing TA-Lib for technical analysis...
python -m pip install TA-Lib
if errorlevel 1 (
    echo WARNING: TA-Lib installation failed
    echo You may need to install it manually or use ta-lib alternative
    python -m pip install ta
)

echo.
echo [7/8] Installing project-specific packages...
python -m pip install quantconnect-stubs databento databento-dbn
if errorlevel 1 (
    echo WARNING: Some project packages failed to install
)

echo.
echo [8/8] Saving installed package list...
python -m pip list > installed_packages.txt
echo Package list saved to installed_packages.txt

echo.
echo =========================================================
echo ENVIRONMENT SETUP COMPLETE!
echo =========================================================
echo.
echo Next steps:
echo 1. Run: python verify_setup.py
echo 2. If verification passes, run: python train_full.py
echo.
echo Virtual environment location: %CD%\venv
echo To activate manually: call venv\Scripts\activate.bat
echo To deactivate: deactivate
echo.
echo =========================================================
pause