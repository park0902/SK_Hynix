import unittest
from src import util, Preprocessing
import os
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocessingTest(unittest.TestCase):
    data_file = '../data/income/eval.csv'
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    def setUp(self):
        pass
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        train_arr, train_label_arr = util.load_data(PreprocessingTest.data_file, num_cols, cate_cols, label_col=label_col)
        Preprocessing.X_train, Preprocessing.X_test, Preprocessing.y_train, Preprocessing.y_train = train_test_split(train_arr, train_label_arr, train_size=0.7)

    def tearDown(self):
        try:
            os.remove('./col_%s.pkl')
        except:
            pass

    def test_prep_missing_val(self):
        X_train = np.array([[np.nan, 1], [2,3], [np.nan, np.nan]])
        X_test = np.array([[np.nan, 1], [2,3]])
        y_train = np.array([1,2,3])
        y_test = np.array([4,5])
        pre_X_train, pre_X_test, pre_y_train, pre_y_train = Preprocessing.prep_missing_val(X_train, X_test, y_train, y_test, mode='remove')
        print()
        print('test_prep_missing_val ==========================')
        print('X_train => ', X_train, 'y_train => ', y_train)
        print('pre_X_train => ', pre_X_train, 'pre_X_test => ', pre_X_test)

    def test_standardization(self):
        X_train = np.array([[1,2], [2,4], [3,6]])
        X_test = np.array([[4,8], [5,10]])
        pre_X_train, pre_X_test, _ = Preprocessing.standardization(X_train, X_test, mode='zscore')
        print()
        print("test_standardization ===========================")
        print('X_train => \n', X_train)
        print('X_test => \n', X_test)

if __name__ == '__main__':
    unittest.main()