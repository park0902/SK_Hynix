
import os
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import unittest
from feature_selection import Lasso
import util
import sklearn.preprocessing
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
class LassoTest(unittest.TestCase):

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
        data, labels, LassoTest.feature_names = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        LassoTest.X_train, LassoTest.X_test, \
        LassoTest.y_train, LassoTest.y_test = sklearn.model_selection.train_test_split(data[0], labels[0], train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_fit(self):
        model = Lasso()
        # params = {'max_iter': 5000, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        params = {}
        model.fit(LassoTest.X_train, labels= LassoTest.y_train, params=params)
        print("test_fit =============================")
        print(model.model)
        return model

    def test_transform(self):
        model = self.test_fit()
        y_predict = model.transform(LassoTest.X_test)
        print("test_transform =============================")
        print(y_predict)

    def test_get_n_components(self):
        model = self.test_fit()
        n_components = model.get_n_components()
        print("test_get_n_components =============================")
        print(n_components)

    def test_get_selected_index_list(self):
        model = self.test_fit()
        selected_index_list = model.get_selected_index_list()
        print("test_get_selected_index_list =============================")
        print(selected_index_list)

    def test_get_selected_feature_names(self):
        model = self.test_fit()
        selected_index_list = model.get_selected_index_list()
        selected_feature_names = model.get_selected_feature_names(LassoTest.feature_names, selected_index_list)
        print("test_get_selected_feature_names =============================")
        print(selected_feature_names)

    def test_save_model(self):
        model = self.test_fit()
        model.save_model('./Lasso.pkl')
        print("test_save_model =============================")

    def test_load_model(self):
        model = Lasso()
        model.load_model('./Lasso.pkl')
        print("test_load_model =============================")


if __name__ == '__main__':
    unittest.main()