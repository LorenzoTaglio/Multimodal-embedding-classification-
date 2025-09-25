from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .xgb_wrapper import XGBWrapper
from ..utils.metrics import metrics
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

class EarlyFusionPipeline:
    def __init__(self, txt_emb, img_emb, y):
        encoder = LabelEncoder()
        self.X = np.concatenate([txt_emb, img_emb], axis=1)
        self.y = encoder.fit_transform(y)
        self.label_names = np.unique(y.tolist())
        self.pca = PCA(n_components=300, random_state=42)
    
    def early_fusion_stratified(self, show_results = True):
        results = metrics(self.label_names)
        xgb_wrapper = XGBWrapper(self.y)
        k_fold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        i = 1
        for train_idx, test_idx in k_fold.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            # X_train, X_test = self.pca.fit_transform(self.X[train_idx]), self.pca.transform(self.X[test_idx])
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            print(f"Fold {i}:")
            _, y_pred, _ = xgb_wrapper.train_test_base(X_train, y_train, X_test)
            label_encoder = LabelEncoder()
            label_encoder.fit(self.label_names)
            y_test_names = label_encoder.inverse_transform(y_test)
            y_pred_names = label_encoder.inverse_transform(y_pred)
            
            results.compute(y_test_names, y_pred_names)
            i+=1
        
        if show_results:
            with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
                print(results.to_dataframe())
                
        return results