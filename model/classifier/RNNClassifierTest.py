
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import unittest
from model.classifier import RNNClassifier
from util import util
from sklearn.model_selection import train_test_split
class RNNClassifierTest(unittest.TestCase):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Fixture
    def setUp(self):
        # file_path = '../../../data/income/eval.data'
        # num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        # cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        # label_col = 'income_bracket'
        # data, labels = util.load_data(file_path, num_cols, cate_cols, label_col=label_col)
        #
        # RNNClassifierTest.X_train, RNNClassifierTest.X_test, \
        # RNNClassifierTest.y_train, RNNClassifierTest.y_test = train_test_split(data, labels, train_size=0.7)

        file_path = '../../../data/fraud/PS_20174392719_1491204439457_log.csv'
        num_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        cate_cols = ['type']
        label_col = 'isFraud'
        data, labels, _ = util.load_data_balance(file_path, num_cols, cate_cols, label_col=label_col)
        RNNClassifierTest.X_train, RNNClassifierTest.X_test, \
        RNNClassifierTest.y_train, RNNClassifierTest.y_test = train_test_split(data[0],
                                                                                                       labels[0],
                                                                                                       train_size=0.7)

    def tearDown(self):
        try:
            # os.remove(MyUtilTest.testfile)
            pass
        except:
            pass

    def test_train(self):
        model = RNNClassifier()
        params = {'n_classes': 2, 'learning_rate': 0.001, 'num_epochs': 4, 'hidden_layers': {'n_layers': 1, 'layers': [{'num_units': 50, 'output_keep_prob': 0.8}]}, 'last_dense_layer_dropout_rate': 0.35000000000000003}
        model.train(RNNClassifierTest.X_train, labels= RNNClassifierTest.y_train, params=params)
        print("test_train =============================")
        print(model.model)
        return model
    #
    # def test_predict(self):
    #     model = self.test_train()
    #     y_predict = model.predict(RNNClassifierTest.X_test)
    #     print("test_predict =============================")
    #     print(y_predict)

    def test_evaluate(self):
        model = self.test_train()
        evl, y_predict = model.evaluate(RNNClassifierTest.X_test, RNNClassifierTest.y_test)
        print("test_evaluate =============================")
        print(evl)

    # def test_get_default_space(self):
    #     space = RNNClassifier.get_default_space(num_epochs=3)
    #     print("test_get_default_space =============================")
    #     print(space)
    #     return space

    # def test_parsing_tune_result(self):
    #     pass
    #     best={'hidden_layers': 0, 'last_dense_layer_dropout_rate': 3, 'num_units_1_1': 4, 'output_keep_prob_1_1': 12}
    #     params = RNNClassifier.parsing_tune_result(best)
    #     print("test_get_default_space =============================")
    #     print(params) # {'n_classes': 2, 'learning_rate': 0.001, 'num_epochs': 4, 'hidden_layers': {'n_layers': 1, 'layers': [{'num_units': 50, 'output_keep_prob': 0.8}]}, 'last_dense_layer_dropout_rate': 0.35000000000000003}
    #     return params

    # def test_tuning(self):
    #     space = RNNClassifier.get_default_space(num_epochs=3)
    #     max_evals = 200
    #     best, trials = RNNClassifier.tuning(RNNClassifierTest.X_train, RNNClassifierTest.y_train, space, max_evals, cv=3)
    #     print("test_tuning =============================")
    #     print(best) #{'hidden_layers': 0, 'last_dense_layer_dropout_rate': 3, 'num_units_1_1': 4, 'output_keep_prob_1_1': 12}

if __name__ == '__main__':
    unittest.main()