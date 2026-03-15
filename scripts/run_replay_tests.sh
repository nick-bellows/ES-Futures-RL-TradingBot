#!/bin/bash

echo "Tradovate Replay Backtest Runner"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "models/best/ppo/ppo_best/best_model.zip" ]; then
    echo "Warning: Default model path not found."
    echo "Please ensure your model is trained and available."
fi

# Run the replay tests
echo "Starting replay backtests..."
python main_replay_test.py "$@"

echo "Replay tests completed."