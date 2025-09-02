from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd


cui_name_df = pd.read_csv("data\\raw\\cui_mapping.csv").set_index("CUI")
def get_cui_name(label):
    return cui_name_df.loc[label]["Canonical name"]



class metrics:
    def __init__(self, labels):
        self.labels = labels
        self.precision = []
        self.recall = []
        self.f1 = []
        self.support = []
         

    def compute(self, y_true, y_pred):
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.labels
        )
        
        self.precision.append(prec)
        self.recall.append(rec)
        self.f1.append(f1)
        self.support.append(support)
    
    def get_label(label):
        return cui_name_df.loc[label]
        
    
    def to_dataframe(self):
        return pd.DataFrame({
            "class": self.labels,
            "class_name": np.array(get_cui_name(label) for label in self.labels),
            "precision_mean": np.mean(self.precision, axis=0),
            "precision_std": np.std(self.precision, axis=0),
            "recall_mean": np.mean(self.recall, axis=0),
            "recall_std": np.std(self.recall, axis=0),
            "f1_mean": np.mean(self.f1, axis=0),
            "f1_std": np.std(self.f1, axis=0),
            "support_mean": np.mean(self.support, axis=0),  # utile per contare i campioni mediamente visti
        })
        