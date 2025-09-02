from sklearn.linear_model import LogisticRegression
import time
import numpy as np

class MetaClassifierWrapper:
    def __init__(self, X_train, X_test, y):
        self.classifier = LogisticRegression(max_iter=1000)
        self.X_train = X_train
        self.X_test = X_test
        self.y = y

    def train_test_base(self):
        time_start = time.time()
        print("\tTraining meta classifier...")
        self.classifier.fit(self.X_train, self.y)    
        print("\tPredicting...")
        y_pred_proba = self.classifier.predict_proba(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        elapsed = time.time() - time_start
        print(f"Time taken: {elapsed:.2f}")
        
        return y_pred
    