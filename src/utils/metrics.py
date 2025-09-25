from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
        self.accuracy = []
         

    def compute(self, y_true, y_pred):
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.labels
        )
        
        acc = accuracy_score(y_true, y_pred)
        
        self.precision.append(prec)
        self.recall.append(rec)
        self.f1.append(f1)
        self.support.append(support)
        self.accuracy.append(acc)
        
    
    def get_label(label):
        return cui_name_df.loc[label]
        
    
    def to_dataframe(self):
        dataframe = pd.DataFrame({
            "class": self.labels,
            "class_name": np.array(get_cui_name(label) for label in self.labels),
            "precision_mean": np.mean(self.precision, axis=0),
            "precision_std": np.std(self.precision, axis=0),
            "recall_mean": np.mean(self.recall, axis=0),
            "recall_std": np.std(self.recall, axis=0),
            "f1_mean": np.mean(self.f1, axis=0),
            "f1_std": np.std(self.f1, axis=0),
            "support_mean": np.mean(self.support, axis=0),  # utile per contare i campioni mediamente visti
            "accuracy_mean": np.mean(self.accuracy),
            "accuracy_std": np.std(self.accuracy)
        })
        
        dataframe.loc[1:, ["accuracy_mean", "accuracy_std"]] = None
        overall = pd.DataFrame({
            "class": ["OVERRALL"],
            "class_name": [None],
            "precision_mean": [np.mean(dataframe["precision_mean"])],
            "precision_std": [np.mean(dataframe["precision_std"])],
            "recall_mean": [np.mean(dataframe["recall_mean"])],
            "recall_std": [np.mean(dataframe["recall_std"])],
            "f1_mean": [np.mean(dataframe["f1_mean"])],
            "f1_std": [np.mean(dataframe["f1_std"])],
            "support_mean": [np.mean(dataframe["support_mean"])],
            "accuracy_mean": [np.mean(self.accuracy)],
            "accuracy_std": [np.std(self.accuracy)],
        })
        
        
        return pd.concat([dataframe, overall], ignore_index=True)
        
    @staticmethod
    def get_accuracy(y_pred,y_true):
        return accuracy_score(y_true, y_pred)
        