import numpy as np
from sklearn.externals import joblib
from abc import *

class FeatureSelect(metaclass=ABCMeta):
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, data, labels=[], params={}):
        pass
    
    @abstractmethod
    def transform(self, data):
        pass
    
    @abstractmethod
    def get_n_components(self):
        pass
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        
    def load_model(self, file_path):
        self.model -= joblib.load(file_path)
        
    @staticmethod
    def get_selected_feature_names(feature_names, selected_ids):
        feature_names = np.array(feature_names)
        return feature_names[selected_ids]
    
if __name__ == '__main__':
    pass

