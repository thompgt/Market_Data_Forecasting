# Market Data Forecasting Pipeline 📈

This repository contains a refactored and modularized pipeline for market data forecasting, originally inspired by the Jane Street Real-Time Market Data Forecasting competition.

## 🚀 Overview

The project provides an end-to-end workflow for:
1.  **Data Acquisition**: Robust fetching with local Jane Street data support and automated fallback to `yfinance` for live market data.
2.  **Feature Engineering**: Automated generation of temporal (rolling mean/std) and lag features.
3.  **Advanced Analysis**: Hierarchical clustering-based feature selection to reduce redundancy and handle multi-collinearity.
4.  **Model Training**: Hyperparameter-tuned Gradient Boosting (LightGBM) models with automated scaling and cross-validation.
5.  **Visualization**: Interactive dashboard built with **Solara** for real-time analysis and performance tracking.

## 📁 Technical Architecture

- `app/data_fetcher.py`: Manages data ingestion and provides fallback mechanisms.
- `app/models.py`: Core `MarketPredictor` class for feature prep and model lifecycle.
- `app/analyzer.py`: `CorrelationAnalyzer` for feature grouping and selection.
- `main.py`: CLI entry point for the full end-to-end pipeline.
- `dashboard.py`: Interactive Solara-based web interface.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📈 Usage

### Run the Pipeline (CLI)
```bash
python main.py
```

### Launch the Dashboard (UI)
```bash
solara run dashboard.py
```

## ⚙️ Technical Workflow

1.  **Ingestion**: Pulls data for a specified ticker (e.g., AAPL).
2.  **Transformation**: Converts raw OHLCV data into 79+ synthetic features and responder targets.
3.  **Clustering**: Features are grouped by complete-linkage hierarchical clustering. Representative features are selected based on target correlation.
4.  **Modeling**: A LightGBM regressor is optimized via `RandomizedSearchCV` to predict next-day returns.
5.  **Evaluation**: Uses a weighted R-squared metric to assess predictive power on unseen test data.
