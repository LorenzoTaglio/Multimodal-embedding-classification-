from sklearn.linear_model import LogisticRegression
import time
import numpy as np

class MetaClassifierWrapper:
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)

    def train_test_base(self, X_train, y_train, X_test):
        time_start = time.time()
        print("\tTraining meta classifier...")
        self.classifier.fit(X_train, y_train)    
        print("\tPredicting...")
        y_pred_proba = self.classifier.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        elapsed = time.time() - time_start
        print(f"Time taken: {elapsed:.2f}")
        
        return y_pred
    