@echo off
echo Tradovate Replay Backtest Runner
echo ================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Error: Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\best\ppo\ppo_best\best_model.zip" (
    echo Warning: Default model path not found.
    echo Please ensure your model is trained and available.
)

REM Run the replay tests
echo Starting replay backtests...
python main_replay_test.py %*

REM Keep window open
pause