
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import unittest
from model.classifier.SVMClassifier import SVMClassifier
from util import util
from sklearn.model_selection import train_test_split
class SVMClassifierTest(unittest.TestCase):
    X_train = []
    X_test = []
    y_train = []
    y_test = []


    # Fixture
    def setUp(self):
        file_path = '../../data/income/eval.data'
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data, labels = util.load_data(file_path, num_cols, cate_cols, label_col=label_col)

        SVMClassifierTest.X_train, SVMClassifierTest.X_test, \
        SVMClassifierTest.y_train, SVMClassifierTest.y_test = train_test_split(data, labels, train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_train(self):
        model = SVMClassifier()
        params = {'max_iter': 100, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        model.train(SVMClassifierTest.X_train, labels= SVMClassifierTest.y_train, params=params)
        print("test_train =============================")
        print(model.model)
        return model

    def test_predict(self):
        model = self.test_train()
        y_predict = model.predict(SVMClassifierTest.X_test)
        print("test_predict =============================")
        print(y_predict)

    def test_evaluate(self):
        model = self.test_train()
        evl, y_predict = model.evaluate(SVMClassifierTest.X_test, SVMClassifierTest.y_test)
        print("test_evaluate =============================")
        print(evl)

    def test_get_default_space(self):
        space = SVMClassifier.get_default_space(max_iter=200)
        print("test_get_default_space =============================")
        print(space)
        return space

    def test_parsing_tune_result(self):
        best={'C_list': 0, 'type': 0}
        params = SVMClassifier.parsing_tune_result(best)
        print("test_get_default_space =============================")
        print(params)
        return params

    # def test_tuning(self):
    #     space = SVMClassifier._space()
    #     max_evals = 1
    #     SVMClassifier.tuning(SVMClassifierTest.X_test, SVMClassifierTest.y_test, space)
    #     evl, y_predict = model.evaluate(SVMClassifierTest.X_test, SVMClassifierTest.y_test)
    #     print("test_evaluate =============================")
    #     print(evl)

if __name__ == '__main__':
    unittest.main()