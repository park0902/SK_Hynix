
import os
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import unittest
from model.classifier.DTClassifier import DTClassifier
import util
import sklearn.preprocessing
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
class DTClassifierTest(unittest.TestCase):

    X_train = []
    X_test = []
    y_train = []
    y_test = []


    # Fixture
    def setUp(self):
        file_path = '../../../data/income/eval.data'
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data, labels = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        DTClassifierTest.X_train, DTClassifierTest.X_test, \
        DTClassifierTest.y_train, DTClassifierTest.y_test = sklearn.model_selection.train_test_split(data[0], labels[0], train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_train(self):
        model = DTClassifier()
        # params = {'max_iter': 5000, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        params =  {}
        model.train(DTClassifierTest.X_train, labels= DTClassifierTest.y_train, params=params)
        print("test_train =============================")
        print(model.model)
        return model

    def test_predict(self):
        model = self.test_train()
        y_predict = model.predict(DTClassifierTest.X_test)
        print("test_predict =============================")
        print(y_predict)

    def test_evaluate(self):
        model = self.test_train()
        evl, y_predict = model.evaluate(DTClassifierTest.X_test, DTClassifierTest.y_test)
        print("test_evaluate =============================")
        print(evl)

    def test_get_default_space(self):
        space = DTClassifier.get_default_space(max_iter=200)
        print("test_get_default_space =============================")
        print(space)
        return space
    #
    def test_parsing_tune_result(self):
        best = {}
        params = DTClassifier.parsing_tune_result(best)
        print("test_get_default_space =============================")
        print(params)
        return params

    def test_tuning(self):
        space = DTClassifier.get_default_space()
        max_evals = 10
        params = {'max_iter':5000}
        best, trials = DTClassifier.tuning(DTClassifierTest.X_test, DTClassifierTest.y_test, space, max_evals, params=params)
        # evl, y_predict = model.evaluate(SVMClassifierTest.X_test, SVMClassifierTest.y_test)
        print("test_tuning =============================")
        print(best)

if __name__ == '__main__':
    unittest.main()