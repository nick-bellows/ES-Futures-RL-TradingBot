@echo off
echo Creating conda environment 'es-rl-bot' with Python 3.10...
call conda create -n es-rl-bot python=3.10 -y
call conda activate es-rl-bot

echo Installing PyTorch with CUDA support for RTX 4080...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo Installing core packages...
pip install pandas==2.1.3 numpy==1.24.3 databento==0.30.0 yfinance==0.2.28

echo Installing QuantConnect packages...
pip install lean==1.18.0 quantconnect-stubs==15000

echo Installing ML packages...
pip install stable-baselines3==2.1.0 gymnasium==0.29.1 tensorboard==2.15.1

echo Installing technical analysis packages...
pip install pandas-ta==0.3.14

echo Installing utilities...
pip install pyyaml==6.0.1 python-dotenv==1.0.0 joblib==1.3.2 scipy==1.11.4 scikit-learn==1.3.2

echo Installing visualization packages...
pip install matplotlib==3.8.1 seaborn==0.13.0 plotly==5.18.0

echo Installing development tools...
pip install pytest==7.4.3 pytest-cov==4.1.0 black==23.11.0 flake8==6.1.0 mypy==1.7.0

echo Installing logging packages...
pip install loguru==0.7.2 wandb==0.16.0 mlflow==2.8.1

echo.
echo ??  IMPORTANT: TA-Lib Installation Required
echo Please download the appropriate TA-Lib wheel for your Python version from:
echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo.
echo For Python 3.10 64-bit, download: TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
echo Then install with: pip install [downloaded_file.whl]
echo.
pause
