from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .xgb_wrapper import XGBWrapper
from ..utils.metrics import metrics
import pandas as pd
import numpy as np

from .meta_classifier_wrapper import MetaClassifierWrapper
from sklearn.decomposition import PCA



class LateFusionPipeline:
    def __init__(self, txt_emb, img_emb, y):
        encoder = LabelEncoder()
        self.X_txt = txt_emb
        self.X_img = img_emb
        self.y = encoder.fit_transform(y)
        self.label_names = np.unique(y.tolist())
        self.pca = PCA(n_components=300, random_state=42)
        
    def late_fusion_stratified(self):
        txt_results = metrics(self.label_names)
        img_results = metrics(self.label_names)
        avg_results = metrics(self.label_names)
        weighted_results = metrics(self.label_names)
        meta_classifier_results = metrics(self.label_names)
        
        xgb_wrapper = XGBWrapper(self.y)
        
        k_fold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        i = 1
        for train_idx, test_idx in k_fold.split(self.X_txt, self.y):
            X_txt_train, X_txt_test = self.pca.fit_transform(self.X_txt[train_idx]), self.pca.transform(self.X_txt[test_idx])
            X_img_train, X_img_test = self.pca.fit_transform(self.X_img[train_idx]), self.pca.transform(self.X_img[test_idx])
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            print(f"Fold {i}:")
            label_encoder = LabelEncoder()
            label_encoder.fit(self.label_names)
            y_test_names = label_encoder.inverse_transform(y_test)
            
            y_txt_pred_proba, y_txt_pred,txt_proba_train = xgb_wrapper.train_test_base(X_txt_train, y_train, X_txt_test, pred_train=True)
            y_txt_pred_names = label_encoder.inverse_transform(y_txt_pred)            
            txt_results.compute(y_test_names, y_txt_pred_names)
            
            y_img_pred_proba, y_img_pred, img_proba_train = xgb_wrapper.train_test_base(X_img_train, y_train, X_img_test, pred_train=True)
            y_img_pred_names = label_encoder.inverse_transform(y_img_pred)
            img_results.compute(y_test_names, y_img_pred_names)
            
            y_pred_fusion = (y_txt_pred_proba + y_img_pred_proba) / 2
            y_pred_fusion = np.argmax(y_pred_fusion, axis=1)
            y_pred_fusion_names = label_encoder.inverse_transform(y_pred_fusion)
            avg_results.compute(y_test_names, y_pred_fusion_names)
            
            weight_txt = 0.6
            weight_img = 0.4
            y_pred_fusion_weighted = (weight_txt * y_txt_pred_proba + weight_img * y_img_pred_proba) / (weight_txt + weight_img)
            y_pred_fusion_weighted = np.argmax(y_pred_fusion_weighted, axis=1)
            y_pred_fusion_weighted_names = label_encoder.inverse_transform(y_pred_fusion_weighted)
            weighted_results.compute(y_test_names, y_pred_fusion_weighted_names)
            
            X_meta_train = np.hstack((txt_proba_train, img_proba_train))
            X_meta_test = np.hstack((y_txt_pred_proba, y_img_pred_proba))
            meta_classifier = MetaClassifierWrapper(X_meta_train, X_meta_test, y_train)
            y_pred_meta = meta_classifier.train_test_base()
            y_pred_meta_names = label_encoder.inverse_transform(y_pred_meta)
            meta_classifier_results.compute(y_test_names, y_pred_meta_names)
            
            i += 1
            
        return {"txt": txt_results, "img":img_results, "avg": avg_results, "weighted": weighted_results, "meta": meta_classifier_results}
            
            
            
            
            
            
            
