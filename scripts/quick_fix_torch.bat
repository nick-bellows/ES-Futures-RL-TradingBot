@echo off
echo Quick PyTorch CUDA Fix
echo =====================
echo This will reinstall PyTorch with CUDA support only

if not exist "venv" (
    echo ERROR: Virtual environment not found!
    echo Please run fix_environment.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Uninstalling current PyTorch...
python -m pip uninstall torch torchvision torchaudio -y

echo Installing PyTorch with CUDA 12.1...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo CUDA version failed, trying CUDA 11.8...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

if errorlevel 1 (
    echo CUDA versions failed, installing CPU version...
    python -m pip install torch torchvision torchaudio
)

echo Testing PyTorch installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

pause