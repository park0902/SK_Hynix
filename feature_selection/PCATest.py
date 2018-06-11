
import os
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import unittest
from feature_selection import PCA
import util
import sklearn.preprocessing
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
class PCATest(unittest.TestCase):

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    feature_names= []


    # Fixture
    def setUp(self):
        file_path = '../../data/income/eval.data'
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data, labels, PCATest.feature_names = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        PCATest.X_train, PCATest.X_test, \
        PCATest.y_train, PCATest.y_test = sklearn.model_selection.train_test_split(data[0], labels[0], train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_fit(self):
        model = PCA()
        # params = {'max_iter': 5000, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        params = {}
        model.fit(PCATest.X_train, labels= PCATest.y_train, params=params)
        print("test_fit =============================")
        print(model.model)
        return model

    def test_transform(self):
        model = self.test_fit()
        y_predict = model.transform(PCATest.X_test)
        print("test_transform =============================")
        print(y_predict)

    def test_get_n_components(self):
        model = self.test_fit()
        n_components = model.get_n_components()
        print("test_get_n_components =============================")
        print(n_components)

    def test_save_model(self):
        model = self.test_fit()
        model.save_model('./PCA.pkl')
        print("test_save_model =============================")

    def test_load_model(self):
        model = PCA()
        model.load_model('./PCA.pkl')
        print("test_load_model =============================")


if __name__ == '__main__':
    unittest.main()