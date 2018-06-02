from util import time_measure
from feature_selection.abstract.FeatureSelection import FeatureSelect
from sklearn import decomposition

class PCA(FeatureSelect):
    @time_measure
    def fit(self, data, labels=[], params={}):
        n_components = params.get('n_components', None)
        self.model = decomposition.PCA(n_components=n_components, random_state=1)
        self.model.fit(data)
        return self.model

    def transform(self, data):
        if self.model is None:
            print('need to fit ...... ')
            exit()
        return self.model.transform(data)

    def get_n_components(self):
        if self.model is None:
            print('need to fit ...... ')
            exit()
        return self.model.n_components_
