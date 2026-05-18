from app.data_fetcher import fetch_market_data
from app.models import MarketPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Starting Market Data Forecasting Pipeline...")
    
    # 1. Fetch Data
    df = fetch_market_data(ticker="AAPL", period="1y")
    if df.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Split Data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    responder_cols = [f'responder_{i}' for i in range(6)] + [f'responder_{i}' for i in range(7, 9)]
    lags_df = train_df[responder_cols].shift(1)

    # 3. Train Model
    predictor = MarketPredictor()
    print("Training model...")
    predictor.train(train_df, lags_df)

    # 4. Predict & Evaluate
    print("Making predictions...")
    test_lags_df = test_df[responder_cols].shift(1)
    predictions = predictor.predict(test_df, test_lags_df)
    
    # Align actuals
    # Re-prepare test features to get aligned target
    test_prepared = predictor.prepare_features(test_df, test_lags_df)
    actuals = test_prepared[predictor.target]
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    r2 = predictor.evaluate(actuals, predictions)
    print(f"Pipeline complete. Weighted R2 Score: {r2:.4f}")

    # 5. Simple Plot
    plt.figure(figsize=(10, 6))
    plt.plot(actuals.values, label='Actual')
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title("Actual vs Predicted Returns")
    plt.savefig("results.png")
    print("Results saved to results.png")

if __name__ == "__main__":
    main()
