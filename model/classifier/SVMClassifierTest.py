
import os
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import unittest
from model.classifier.SVMClassifier import SVMClassifier
import util
import sklearn.preprocessing
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
class SVMClassifierTest(unittest.TestCase):

    X_train = []
    X_test = []
    y_train = []
    y_test = []


    # Fixture
    def setUp(self):
        # file_path = '../../data/income/eval.data'
        # num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        # cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        # label_col = 'income_bracket'
        file_path = '../../../data/fraud/PS_20174392719_1491204439457_log.csv'
        num_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        cate_cols = ['type']
        label_col = 'isFraud'
        data, labels = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        SVMClassifierTest.X_train, SVMClassifierTest.X_test, \
        SVMClassifierTest.y_train, SVMClassifierTest.y_test = sklearn.model_selection.train_test_split(data[0], labels[0], train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_train(self):
        model = SVMClassifier()
        # params = {'max_iter': 5000, 'type': {'C': 16.0, 'kernel':'rbf', 'gamma':2.0 } }
        params =  {'max_iter': -1, 'type': {'C': 32.0, 'gamma': 0.000244140625, 'kernel': 'rbf'}}
        model.train(SVMClassifierTest.X_train, labels= SVMClassifierTest.y_train, params=params)
        print("test_train =============================")
        print(model.model)
        return model

    # def test_predict(self):
    #     model = self.test_train()
    #     y_predict = model.predict(SVMClassifierTest.X_test)
    #     print("test_predict =============================")
    #     print(y_predict)

    def test_evaluate(self):
        model = self.test_train()
        evl, y_predict = model.evaluate(SVMClassifierTest.X_test, SVMClassifierTest.y_test)
        print("test_evaluate =============================")
        print(evl)

    # def test_get_default_space(self):
    #     space = SVMClassifier.get_default_space(max_iter=200)
    #     print("test_get_default_space =============================")
    #     print(space)
    #     return space
    #
    # def test_parsing_tune_result(self):
    #     best = {'C_2': 4, 'gamma_2': 0, 'type': 0} # {'max_iter': 200, 'type': {'C': 0.5, 'gamma': 3.0517578125e-05, 'kernel': 'rbf'}}
    #     best = {'C_2': 10, 'gamma_2': 3, 'type': 0} # {'max_iter': 200, 'type': {'C': 32.0, 'gamma': 0.000244140625, 'kernel': 'rbf'}}
    #
    #     params = SVMClassifier.parsing_tune_result(best)
    #     print("test_get_default_space =============================")
    #     print(params)
    #     return params

    # def test_tuning(self):
    #     space = SVMClassifier.get_default_space()
    #     max_evals = 10
    #     params = {'max_iter':5000}
    #     best, trials = SVMClassifier.tuning(SVMClassifierTest.X_test, SVMClassifierTest.y_test, space, max_evals, params=params)
    #     # evl, y_predict = model.evaluate(SVMClassifierTest.X_test, SVMClassifierTest.y_test)
    #     print("test_tuning =============================")
    #     print(best)

if __name__ == '__main__':
    unittest.main()