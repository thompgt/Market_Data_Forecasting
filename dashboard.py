import solara
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.data_fetcher import fetch_market_data
from app.models import MarketPredictor
import numpy as np

# Reactive variables
ticker = solara.reactive("AAPL")
data = solara.reactive(pd.DataFrame())
predictions = solara.reactive(np.array([]))
actuals = solara.reactive(pd.Series())
is_loading = solara.reactive(False)
error_msg = solara.reactive("")

def run_pipeline():
    is_loading.set(True)
    error_msg.set("")
    try:
        df = fetch_market_data(ticker=ticker.value, period="1y")
        if df.empty:
            error_msg.set(f"Could not fetch data for {ticker.value}")
            return
        
        data.set(df)
        
        # Simple training for dashboard demo
        predictor = MarketPredictor()
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        responder_cols = [f'responder_{i}' for i in range(6)] + [f'responder_{i}' for i in range(7, 9)]
        lags_df = train_df[responder_cols].shift(1)
        
        predictor.train(train_df, lags_df)
        
        test_lags_df = test_df[responder_cols].shift(1)
        preds = predictor.predict(test_df, test_lags_df)
        
        test_prepared = predictor.prepare_features(test_df, test_lags_df)
        acts = test_prepared[predictor.target]
        
        min_len = min(len(preds), len(acts))
        predictions.set(preds[:min_len])
        actuals.set(acts[:min_len])
        
    except Exception as e:
        error_msg.set(f"Error: {str(e)}")
    finally:
        is_loading.set(False)

@solara.component
def Page():
    with solara.Column(style={"padding": "20px", "max-width": "1200px", "margin": "0 auto"}):
        solara.Title("Market Data Forecasting Dashboard")
        solara.Markdown("# 📈 Market Data Forecasting")
        solara.Markdown("Refactored Jane Street Real-Time Market Data Forecasting Pipeline.")
        
        with solara.Row():
            solara.InputText("Ticker Symbol", value=ticker, continuous_update=False)
            solara.Button("Run Forecast Pipeline", on_click=run_pipeline, color="primary")
            
        if is_loading.value:
            solara.ProgressLinear(True)
            
        if error_msg.value:
            solara.Error(error_msg.value)
            
        if not data.value.empty:
            with solara.GridFixed(columns=2):
                with solara.Card("Historical Data (Returns)"):
                    fig = px.line(data.value, x=data.value.index, y="feature_00", title="Daily Returns (feature_00)")
                    solara.FigurePlotly(fig)
                
                with solara.Card("Feature Correlation"):
                    # Just first 10 for visibility
                    corr = data.value[[f'feature_{i:02d}' for i in range(10)]].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap (First 10 Features)")
                    solara.FigurePlotly(fig)

            if len(predictions.value) > 0:
                with solara.Card("Actual vs Predicted Returns"):
                    plot_df = pd.DataFrame({
                        "Actual": actuals.value.values,
                        "Predicted": predictions.value
                    })
                    fig = px.line(plot_df, title="Forecast Results")
                    solara.FigurePlotly(fig)
                    
                with solara.Row():
                    r2 = 1 - np.sum((actuals.value.values - predictions.value)**2) / np.sum(actuals.value.values**2)
                    solara.Metric(label="Weighted R2 Score", value=f"{r2:.4f}")
                    solara.Metric(label="Test Samples", value=str(len(predictions.value)))

# Required for solara run
app = Page
