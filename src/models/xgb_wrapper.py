from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time


class XGBWrapper:
    def __init__(self, y):
        self.y = y
        
        self.classes = np.unique(self.y)
        self.class_weights = compute_class_weight("balanced", classes=self.classes, y=self.y)
        self.weight_dict = {c: w for c, w in zip(self.classes, self.class_weights)}
        
        self.base_classifier = []
        
        self.optimal_classifier = []
    
    def train_base(self,X_train, y_train):
        self.base_classifier.append(
            XGBClassifier(
                objective="multi:softprob",
                num_class= len(self.classes),  # number of unique labels
                eval_metric="logloss",  
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        )
        
        sample_weights = np.array([self.weight_dict[label] for label in y_train])
        self.base_classifier[-1].fit(X_train, y_train, sample_weight=sample_weights, verbose=True)
    
        
    def predict_base(self, X_test):
        y_pred_proba = self.base_classifier[-1].predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred_proba, y_pred

    def predict_train(self, X_train):
        y_pred_proba_train = self.base_classifier[-1].predict_proba(X_train)
        return y_pred_proba_train
    
    
    def train_test_base(self, X_train, y_train ,X_test, pred_train = False):
        time_start = time.time()
        y_pred_proba_train = None
        print("\tTraining classifier...")
        self.train_base(X_train, y_train)    
        print("\tPredicting...")
        y_pred_proba, y_pred = self.predict_base(X_test)
        if pred_train:
            y_pred_proba_train = self.predict_train(X_train)
        elapsed = time.time() - time_start
        print(f"Time taken: {elapsed:.2f}")
        
        return y_pred_proba, y_pred, y_pred_proba_train
    
    def mean_predictions (self, X_test, y_test_len):
        mean_preds = np.zeros((y_test_len, len(self.classes)))
        
        for classifier in self.base_classifier:
            mean_preds += classifier.predict_proba(X_test)
        
        return mean_preds / len(self.base_classifier)
        