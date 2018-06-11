
import os
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import unittest
from feature_selection import PLS
import util
import sklearn.preprocessing
# from sklearn.model_selection import train_test_split

import sklearn.model_selection
class PLSTest(unittest.TestCase):

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
        data, labels, PLSTest.feature_names = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        PLSTest.X_train, PLSTest.X_test, \
        PLSTest.y_train, PLSTest.y_test = sklearn.model_selection.train_test_split(data[0], labels[0], train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_fit(self):
        model = PLS()
        # params = {'max_iter': 5000, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        params = {}
        model.fit(PLSTest.X_train, labels= PLSTest.y_train, params=params)
        print("test_fit =============================")
        print(model.model)
        return model

    def test_transform(self):
        model = self.test_load_model()
        y_predict = model.transform(PLSTest.X_test)
        print("test_transform =============================")
        print(y_predict)

    def test_get_n_components(self):
        model = self.test_load_model()
        n_components = model.get_n_components()
        print("test_get_n_components =============================")
        print(n_components)

    def test_save_model(self):
        model = self.test_fit()
        model.save_model('./PLS.pkl')
        print("test_save_model =============================")

    def test_load_model(self):
        model = PLS()
        model.load_model('./PLS.pkl')
        print("test_load_model =============================")
        return model


if __name__ == '__main__':
    unittest.main()