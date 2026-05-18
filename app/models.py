import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from .analyzer import CorrelationAnalyzer

class MarketPredictor:
    def __init__(self):
        self.feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.target = 'responder_6'
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols_final = None
        
    def create_temporal_features(self, df, symbol_id):
        windows = [5, 10, 20]
        temp_df = df.copy()
        symbol_mask = temp_df['symbol_id'] == symbol_id
        symbol_data = temp_df[symbol_mask]
        
        feature_dfs = []
        for feat in self.feature_cols:
            if feat not in symbol_data.columns: continue
            feature_series = symbol_data[feat]
            for window in windows:
                window_df = pd.DataFrame({
                    f'{feat}_mean_{window}': feature_series.rolling(window, min_periods=1).mean(),
                    f'{feat}_std_{window}': feature_series.rolling(window, min_periods=1).std()
                }, index=symbol_data.index)
                feature_dfs.append(window_df)
        
        if feature_dfs:
            all_features = pd.concat(feature_dfs, axis=1)
            temp_df.loc[symbol_mask, all_features.columns] = all_features
        return temp_df

    def create_lag_features(self, df, symbol_id, lags=[1, 2, 3]):
        temp_df = df.copy()
        symbol_mask = temp_df['symbol_id'] == symbol_id
        symbol_data = temp_df[symbol_mask]
        
        feature_dfs = []
        for feat in self.feature_cols:
            if feat not in symbol_data.columns: continue
            feature_series = symbol_data[feat]
            lag_dict = {f'{feat}_lag_{lag}': feature_series.shift(lag) for lag in lags}
            feature_dfs.append(pd.DataFrame(lag_dict, index=symbol_data.index))
        
        if feature_dfs:
            all_features = pd.concat(feature_dfs, axis=1)
            temp_df.loc[symbol_mask, all_features.columns] = all_features
        return temp_df

    def prepare_features(self, df, lags_df=None):
        temp_df = df.copy()
        if lags_df is not None:
            for resp in [f'responder_{i}' for i in range(6)] + [f'responder_{i}' for i in range(7, 9)]:
                if resp in lags_df.columns:
                    temp_df[f'{resp}_prev'] = lags_df[resp]
        
        for symbol in temp_df['symbol_id'].unique():
            temp_df = self.create_temporal_features(temp_df, symbol)
            temp_df = self.create_lag_features(temp_df, symbol)
        
        feature_cols = self.get_feature_columns(temp_df)
        for col in feature_cols:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].ffill().fillna(0)
        
        return temp_df.dropna(subset=[self.target])

    def get_feature_columns(self, df):
        return [col for col in df.columns 
               if col.startswith('feature_') or 
                  col.endswith(('_mean_5', '_mean_10', '_mean_20',
                               '_std_5', '_std_10', '_std_20',
                               '_lag_1', '_lag_2', '_lag_3')) or
                  col.endswith('_prev')]

    def train(self, train_df, lags_df=None):
        df = self.prepare_features(train_df, lags_df)
        correlation_analyzer = CorrelationAnalyzer()
        feature_cols = self.get_feature_columns(df)
        correlation_analyzer.find_feature_groups(df, feature_cols)
        self.feature_cols_final = correlation_analyzer.select_representative_features(df, self.target)

        X = df[self.feature_cols_final]
        y = df[self.target]
        X_scaled = self.scaler.fit_transform(X)
        
        base_model = LGBMRegressor(random_state=42, verbose=-1)
        param_distributions = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.1),
            'max_depth': randint(3, 8)
        }
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=10,
            cv=3,
            n_jobs=1,
            random_state=42
        )
        random_search.fit(X_scaled, y)
        self.models['ensemble'] = random_search.best_estimator_
        return self.models['ensemble']

    def predict(self, test_df, lags_df=None):
        df = self.prepare_features(test_df, lags_df)
        X = df[self.feature_cols_final]
        X_scaled = self.scaler.transform(X)
        return self.models['ensemble'].predict(X_scaled)

    def evaluate(self, y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones(len(y_true)) / len(y_true)
        weighted_mse = np.sum(weights * (y_true - y_pred) ** 2)
        weighted_var = np.sum(weights * y_true ** 2)
        return 1 - weighted_mse / weighted_var if weighted_var != 0 else 0
