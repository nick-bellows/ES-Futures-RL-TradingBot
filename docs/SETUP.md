# Environment Setup Guide

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (optional, for faster training)
- NinjaTrader 8 (for live/paper trading)

## Quick Setup

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### CUDA / GPU Setup

Install PyTorch with CUDA for GPU-accelerated training:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is available:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Training performance:
- **CPU**: ~1-2 hours for 500K timesteps
- **GPU (RTX 4080)**: ~10-20 minutes for 500K timesteps

### TA-Lib (Technical Analysis)

Download the appropriate wheel from the [TA-Lib archive](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), then:
```bash
pip install TA_Lib-<version>.whl
```

## Configuration

1. Copy `.env.template` to `.env` and fill in your API keys
2. Edit `config/trading_config.yaml` for trading parameters
3. Edit `ninjatrader_config.json` for NinjaTrader connection settings

## Package Installation Order

If you encounter dependency conflicts, install in this order:

1. **Core**: `numpy`, `pandas`
2. **PyTorch**: `torch`, `torchvision`, `torchaudio` (with CUDA)
3. **Scientific**: `scipy`, `scikit-learn`, `matplotlib`
4. **RL/ML**: `stable-baselines3`, `gymnasium`
5. **Utilities**: `tensorboard`, `pyyaml`, `python-dotenv`
6. **Trading**: `TA-Lib`, `databento`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named pandas._libs` | Reinstall pandas: `pip install --force-reinstall pandas` |
| DLL errors with torch | Install CUDA version: `pip install torch --extra-index-url ...` |
| CUDA not available | Verify NVIDIA drivers, reinstall PyTorch with CUDA |
| NumPy version conflicts | Pin numpy: `pip install numpy==1.26.4` |

## Verification

After setup, verify all components:
```bash
python -c "
import torch, numpy, pandas, gymnasium, stable_baselines3
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'NumPy: {numpy.__version__}')
print('All imports successful')
"
```
