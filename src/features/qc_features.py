"""
QuantConnect-Compatible Feature Engineering Pipeline

Creates 47 technical indicators compatible with QuantConnect's indicator library.
Features are designed to match QC's implementation for seamless deployment.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QCFeatureEngine:
    """Feature engineering pipeline with QuantConnect-compatible indicators"""
    
    def __init__(self, lookback_period: int = 200):
        self.lookback_period = lookback_period
        self.feature_names = []
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 47 QC-compatible features"""
        logger.info("Calculating 47 QuantConnect-compatible features")
        
        # Copy DataFrame to avoid modifying original
        features_df = df.copy()
        
        # Convert OHLCV to numpy arrays for TA-Lib (ensure float64)
        open_prices = df['Open'].astype(np.float64).values
        high_prices = df['High'].astype(np.float64).values  
        low_prices = df['Low'].astype(np.float64).values
        close_prices = df['Close'].astype(np.float64).values
        volume = df['Volume'].astype(np.float64).values
        
        # Price-based features (10 features)
        features_df = self._add_price_features(features_df, open_prices, high_prices, low_prices, close_prices)
        
        # Moving averages (8 features)
        features_df = self._add_moving_averages(features_df, close_prices)
        
        # Momentum indicators (12 features)
        features_df = self._add_momentum_indicators(features_df, open_prices, high_prices, low_prices, close_prices, volume)
        
        # Volatility indicators (5 features)
        features_df = self._add_volatility_indicators(features_df, high_prices, low_prices, close_prices)
        
        # Volume indicators (4 features)
        features_df = self._add_volume_indicators(features_df, high_prices, low_prices, close_prices, volume)
        
        # Pattern recognition (5 features)
        features_df = self._add_pattern_features(features_df, open_prices, high_prices, low_prices, close_prices)
        
        # Statistical features (3 features)
        features_df = self._add_statistical_features(features_df, close_prices)
        
        # Drop rows with NaN values (from initial calculation periods)
        features_df = features_df.dropna().reset_index(drop=True)
        
        logger.info(f"Feature engineering complete. Dataset shape: {features_df.shape}")
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame, open_p, high_p, low_p, close_p) -> pd.DataFrame:
        """Add price-based features (10 features)"""
        # 1-4. OHLC normalized by previous close
        df['open_norm'] = open_p / np.roll(close_p, 1) - 1
        df['high_norm'] = high_p / np.roll(close_p, 1) - 1  
        df['low_norm'] = low_p / np.roll(close_p, 1) - 1
        df['close_norm'] = close_p / np.roll(close_p, 1) - 1
        
        # 5. Typical price
        df['typical_price'] = (high_p + low_p + close_p) / 3
        
        # 6. Weighted close
        df['weighted_close'] = (high_p + low_p + 2 * close_p) / 4
        
        # 7. Price range
        df['price_range'] = (high_p - low_p) / close_p
        
        # 8. Gap from previous close
        df['gap'] = (open_p - np.roll(close_p, 1)) / np.roll(close_p, 1)
        
        # 9. Intraday return
        df['intraday_return'] = (close_p - open_p) / open_p
        
        # 10. Log return
        df['log_return'] = np.log(close_p / np.roll(close_p, 1))
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame, close_p) -> pd.DataFrame:
        """Add moving average features (8 features)"""
        # 11-14. Simple Moving Averages
        df['sma_5'] = talib.SMA(close_p, timeperiod=5) / close_p - 1
        df['sma_20'] = talib.SMA(close_p, timeperiod=20) / close_p - 1
        df['sma_50'] = talib.SMA(close_p, timeperiod=50) / close_p - 1
        df['sma_200'] = talib.SMA(close_p, timeperiod=200) / close_p - 1
        
        # 15-16. Exponential Moving Averages
        df['ema_12'] = talib.EMA(close_p, timeperiod=12) / close_p - 1
        df['ema_26'] = talib.EMA(close_p, timeperiod=26) / close_p - 1
        
        # 17. MACD line (normalized)
        macd, _, _ = talib.MACD(close_p)
        df['macd'] = macd / close_p
        
        # 18. Moving average convergence
        df['ma_convergence'] = (talib.SMA(close_p, 20) - talib.SMA(close_p, 50)) / close_p
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame, open_p, high_p, low_p, close_p, volume=None) -> pd.DataFrame:
        """Add momentum indicators (12 features)"""
        # 19-21. RSI with different periods
        df['rsi_14'] = talib.RSI(close_p, timeperiod=14) / 100  # Normalize to 0-1
        df['rsi_7'] = talib.RSI(close_p, timeperiod=7) / 100
        df['rsi_21'] = talib.RSI(close_p, timeperiod=21) / 100
        
        # 22-23. Stochastic oscillator
        slowk, slowd = talib.STOCH(high_p, low_p, close_p)
        df['stoch_k'] = slowk / 100
        df['stoch_d'] = slowd / 100
        
        # 24. Williams %R
        df['williams_r'] = (talib.WILLR(high_p, low_p, close_p) + 100) / 100  # Normalize to 0-1
        
        # 25. Commodity Channel Index
        df['cci'] = talib.CCI(high_p, low_p, close_p) / 200  # Normalize
        
        # 26. Money Flow Index
        if volume is not None:
            df['mfi'] = talib.MFI(high_p, low_p, close_p, volume, timeperiod=14) / 100
        else:
            df['mfi'] = 0.5  # Default neutral value
        
        # 27. Rate of Change
        df['roc'] = talib.ROC(close_p, timeperiod=10) / 100
        
        # 28. Average Directional Index
        df['adx'] = talib.ADX(high_p, low_p, close_p, timeperiod=14) / 100
        
        # 29. Plus Directional Indicator
        df['plus_di'] = talib.PLUS_DI(high_p, low_p, close_p, timeperiod=14) / 100
        
        # 30. Minus Directional Indicator  
        df['minus_di'] = talib.MINUS_DI(high_p, low_p, close_p, timeperiod=14) / 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame, high_p, low_p, close_p) -> pd.DataFrame:
        """Add volatility indicators (5 features)"""
        # 31-33. Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_p)
        df['bb_upper'] = (bb_upper - close_p) / close_p
        df['bb_lower'] = (bb_lower - close_p) / close_p
        df['bb_width'] = (bb_upper - bb_lower) / close_p
        
        # 34. Average True Range
        df['atr'] = talib.ATR(high_p, low_p, close_p, timeperiod=14) / close_p
        
        # 35. True Range
        df['true_range'] = talib.TRANGE(high_p, low_p, close_p) / close_p
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame, high_p, low_p, close_p, volume) -> pd.DataFrame:
        """Add volume indicators (4 features)"""
        # 36. On-Balance Volume (normalized by price)
        df['obv'] = talib.OBV(close_p, volume) / (close_p * 1000)
        
        # 37. Volume Rate of Change
        df['volume_roc'] = talib.ROC(volume, timeperiod=10) / 100
        
        # 38. Accumulation/Distribution Line
        df['ad'] = talib.AD(high_p, low_p, close_p, volume) / (close_p * volume.mean())
        
        # 39. Chaikin A/D Oscillator
        df['adosc'] = talib.ADOSC(high_p, low_p, close_p, volume) / (close_p * 100)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame, open_p, high_p, low_p, close_p) -> pd.DataFrame:
        """Add pattern recognition features (5 features)"""
        # 40. Doji pattern
        df['doji'] = talib.CDLDOJI(open_p, high_p, low_p, close_p) / 100
        
        # 41. Hammer pattern
        df['hammer'] = talib.CDLHAMMER(open_p, high_p, low_p, close_p) / 100
        
        # 42. Engulfing pattern
        df['engulfing'] = talib.CDLENGULFING(open_p, high_p, low_p, close_p) / 100
        
        # 43. Morning star pattern
        df['morning_star'] = talib.CDLMORNINGSTAR(open_p, high_p, low_p, close_p) / 100
        
        # 44. Evening star pattern  
        df['evening_star'] = talib.CDLEVENINGSTAR(open_p, high_p, low_p, close_p) / 100
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame, close_p) -> pd.DataFrame:
        """Add statistical features (3 features)"""
        # 45. Rolling correlation with volume
        df['price_volume_corr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # 46. Rolling standard deviation
        df['rolling_std'] = df['Close'].rolling(20).std() / df['Close']
        
        # 47. Z-score (price relative to 20-period mean)
        rolling_mean = df['Close'].rolling(20).mean()
        rolling_std = df['Close'].rolling(20).std()
        df['z_score'] = (df['Close'] - rolling_mean) / rolling_std
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return [
            # Price features (10)
            'open_norm', 'high_norm', 'low_norm', 'close_norm', 'typical_price',
            'weighted_close', 'price_range', 'gap', 'intraday_return', 'log_return',
            
            # Moving averages (8)  
            'sma_5', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'macd', 'ma_convergence',
            
            # Momentum (12)
            'rsi_14', 'rsi_7', 'rsi_21', 'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'mfi', 'roc', 'adx', 'plus_di', 'minus_di',
            
            # Volatility (5)
            'bb_upper', 'bb_lower', 'bb_width', 'atr', 'true_range',
            
            # Volume (4)
            'obv', 'volume_roc', 'ad', 'adosc',
            
            # Patterns (5)
            'doji', 'hammer', 'engulfing', 'morning_star', 'evening_star',
            
            # Statistical (3)
            'price_volume_corr', 'rolling_std', 'z_score'
        ]


def main():
    """Test feature engineering pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    # Load continuous data
    df = pd.read_csv('data/continuous/ES_continuous_1min.csv')
    
    # Initialize feature engine
    engine = QCFeatureEngine()
    
    # Calculate features
    features_df = engine.calculate_all_features(df)
    
    # Save features
    features_df.to_csv('data/processed/ES_features_1min.csv', index=False)
    
    print(f"Features calculated: {len(engine.get_feature_columns())} features")
    print(f"Dataset shape: {features_df.shape}")
    print(f"Date range: {features_df['Time'].min()} to {features_df['Time'].max()}")
    
    return features_df


if __name__ == "__main__":
    main()