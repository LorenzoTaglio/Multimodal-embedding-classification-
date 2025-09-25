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
        
    def weighted_mean(self, txt_pred, img_pred, y_true, step):
        best_score = -np.inf
        best_weights = [0,0]
        for w_t in range(0,1, step):
            w_i = 1-w_t
            y_pred_proba = w_t*txt_pred, w_i*img_pred
            y_pred = np.argmax(y_pred_proba, axis=1)
            score = metrics.get_accuracy(y_pred,y_true)
            if score > best_score:
                best_score = score
                best_weights = [w_t, w_i]
        
        return best_weights[0], best_weights[1]
        
    def late_fusion_stratified(self):
        txt_results = metrics(self.label_names)
        img_results = metrics(self.label_names)
        avg_results = metrics(self.label_names)
        weighted_results = metrics(self.label_names)

        
        xgb_wrapper = XGBWrapper(self.y)
        
        k_fold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        i = 1
        for train_idx, test_idx in k_fold.split(self.X_txt, self.y):
            # X_txt_train, X_txt_test = self.pca.fit_transform(self.X_txt[train_idx]), self.pca.transform(self.X_txt[test_idx])
            # X_img_train, X_img_test = self.pca.fit_transform(self.X_img[train_idx]), self.pca.transform(self.X_img[test_idx])
            X_txt_train, X_txt_test = self.X_txt[train_idx], self.X_txt[test_idx]
            X_img_train, X_img_test = self.X_img[train_idx], self.X_img[test_idx]
            
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
            
            weight_txt, weight_img = self.weighted_mean(y_txt_pred_proba, y_img_pred_proba, y_test, 0.2)
            y_pred_fusion_weighted = (weight_txt * y_txt_pred_proba + weight_img * y_img_pred_proba)
            y_pred_fusion_weighted = np.argmax(y_pred_fusion_weighted, axis=1)
            y_pred_fusion_weighted_names = label_encoder.inverse_transform(y_pred_fusion_weighted)
            weighted_results.compute(y_test_names, y_pred_fusion_weighted_names)
            
            
            i += 1
            
        return {"txt": txt_results, "img":img_results, "avg": avg_results, "weighted": weighted_results}
    
    
    def late_fusion_meta(self):
        meta_results = metrics(self.label_names)

        meta_classifier = MetaClassifierWrapper()
        label_encoder = LabelEncoder()
        label_encoder.fit(self.label_names)
        
        
        n_classes = len(np.unique(self.y))
        
        outer_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        outer_fold_counter = 0
        for train_idx, test_idx in outer_k_fold.split(self.X_txt, self.y):        
            img_classifier = XGBWrapper(self.y)
            txt_classifier = XGBWrapper(self.y)
        
            outer_fold_counter +=1
            
            X_txt_train, X_txt_test = self.X_txt[train_idx], self.X_txt[test_idx]
            X_img_train, X_img_test = self.X_img[train_idx], self.X_img[test_idx]
        
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            y_test_names = label_encoder.inverse_transform(y_test)

            
            # inizialize out of fold predictions
            oof_txt_train = np.zeros((len(y_train), n_classes))
            oof_img_train = np.zeros((len(y_train), n_classes))
            print(f"Outer Fold {outer_fold_counter}")
            # Split train in (train,val)
            inner_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            inner_fold_counter = 0
            for inner_train_idx, inner_val_idx in inner_k_fold.split(X_txt_train, y_train):
                inner_fold_counter +=1
                
                print(f"\tInner fold {inner_fold_counter}")
                
                inner_txt_train, inner_txt_val = X_txt_train[inner_train_idx], X_txt_train[inner_val_idx]
                inner_img_train, inner_img_val = X_img_train[inner_train_idx], X_img_train[inner_val_idx]
                
                inner_y_train, inner_y_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                val_txt_pred_proba, _, _ = txt_classifier.train_test_base(inner_txt_train, inner_y_train, inner_txt_val)
                val_img_pred_proba, _, _ = img_classifier.train_test_base(inner_img_train, inner_y_train, inner_img_val)
            
                oof_txt_train[inner_val_idx] = val_txt_pred_proba
                oof_img_train[inner_val_idx] = val_img_pred_proba

            
            # Train meta classifier
            meta_train = np.concatenate([oof_txt_train, oof_img_train], axis=1)
            meta_test = np.concatenate([txt_classifier.mean_predictions(X_test=X_txt_test, y_test_len=len(y_test)), 
                                        img_classifier.mean_predictions(X_test=X_img_test, y_test_len=len(y_test))], 
                                       axis=1)
            
            y_pred = meta_classifier.train_test_base(meta_train, y_train, meta_test)
            y_pred_names = label_encoder.inverse_transform(y_pred)
            meta_results.compute(y_test_names, y_pred_names)
            
        return meta_results
            
            
            
