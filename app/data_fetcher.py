import pandas as pd
import numpy as np
import yfinance as yf
import os

def fetch_market_data(ticker="AAPL", period="1y", fallback=True):
    """
    Fetch market data. Tries local Jane Street format first, then falls back to yfinance.
    """
    path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
    
    if os.path.exists(path):
        try:
            samples = []
            for i in range(1):
                file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
                if os.path.exists(file_path):
                    part = pd.read_parquet(file_path)
                    samples.append(part)
            if samples:
                df = pd.concat(samples, ignore_index=True)
                print(f"Loaded local data with shape: {df.shape}")
                return df
        except Exception as e:
            print(f"Error loading local data: {e}")

    if fallback:
        print(f"Falling back to yfinance for {ticker}...")
        data = yf.download(ticker, period=period)
        if data.empty:
            return pd.DataFrame()
        
        # Transform yfinance data into a format compatible with the predictor
        # We'll create synthetic features feature_00 to feature_78
        df = data.copy()
        df = df.reset_index()
        
        # Ensure column names are simple strings
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Create target 'responder_6' (next day return)
        df['responder_6'] = df['Close'].pct_change().shift(-1)
        
        # Create symbol_id
        df['symbol_id'] = 0
        
        # Create features 0-78
        # We can use simple indicators or just noise for features we don't have
        for i in range(79):
            if i == 0:
                df[f'feature_{i:02d}'] = df['Close'].pct_change()
            elif i == 1:
                df[f'feature_{i:02d}'] = df['Volume'].pct_change()
            elif i == 2:
                df[f'feature_{i:02d}'] = (df['High'] - df['Low']) / df['Close']
            else:
                # Add some random noise for the rest to keep the pipeline running
                df[f'feature_{i:02d}'] = np.random.normal(0, 1, len(df))
        
        # Add other responders as 0s
        for i in range(6):
            if i != 6:
                df[f'responder_{i}'] = 0
        df['responder_7'] = 0
        df['responder_8'] = 0
        
        df = df.dropna()
        print(f"Generated synthetic market data from yfinance with shape: {df.shape}")
        return df
    
    return pd.DataFrame()
