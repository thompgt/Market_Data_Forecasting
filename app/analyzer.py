import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

class CorrelationAnalyzer:
    def __init__(self, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold
        self.feature_groups = {}
        self.selected_features = []
        
    def calculate_correlations(self, df, feature_cols):
        valid_cols = [col for col in feature_cols if not df[col].isnull().all()]
        if not valid_cols:
            return pd.DataFrame()
        corr = df[valid_cols].corr().fillna(0)
        return corr
    
    def find_feature_groups(self, df, feature_cols):
        try:
            corr = self.calculate_correlations(df, feature_cols)
            if corr.empty: return {}
            
            distance_matrix = 1 - np.abs(corr)
            tri_upper = distance_matrix.values[np.triu_indices(n=distance_matrix.shape[0], k=1)]
            tri_upper = np.nan_to_num(tri_upper, nan=1.0)
            
            linkage = hierarchy.linkage(tri_upper, method='complete')
            clusters = hierarchy.fcluster(linkage, self.correlation_threshold, criterion='distance')
            
            self.feature_groups = {}
            for feature, cluster_id in zip(corr.columns, clusters):
                self.feature_groups.setdefault(cluster_id, []).append(feature)
                
            return self.feature_groups
        except Exception as e:
            print(f"Error in feature grouping: {e}")
            self.feature_groups = {1: feature_cols}
            return self.feature_groups
    
    def select_representative_features(self, df, target):
        self.selected_features = []
        for group in self.feature_groups.values():
            correlations = []
            for feature in group:
                correlation = abs(df[feature].fillna(df[feature].mean()).corr(df[target].fillna(df[target].mean())))
                correlations.append((feature, correlation if np.isfinite(correlation) else 0))
            
            if correlations:
                self.selected_features.append(max(correlations, key=lambda x: x[1])[0])
        return self.selected_features

    def get_correlation_heatmap(self, df, feature_cols):
        corr = self.calculate_correlations(df, feature_cols)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap='RdBu_r', center=0, ax=ax)
        plt.tight_layout()
        return fig
