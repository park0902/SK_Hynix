import unittest
from src import util
import os

class UtilTest(unittest.TestCase):
    data_file ='../data/income/eval.csv'

    def setUp(self):
        pass

    def tearDown(self):
        try:
            os.remove('./col_%s.pkl')
        except:
            pass


    def test_load_data(self):
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data_arr, labels_arr = util.load_data(UtilTest.data_file, num_cols, cate_cols, label_col=label_col)
        print("test load_data =========================")
        print('data_arr => %s' %str(data_arr.shape))
        print('labels_arr => %s' % str(labels_arr.shape))

    def test_load_data_with_name(self):
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data_arr, labels_arr, data_names, labels_names = util.load_data_with_name(UtilTest.data_file, num_cols, cate_cols, label_col=label_col)
        print('test_load_data_with_name ===============')
        print(data_names)
        print(labels_names)

    def test_load_data_balance(self):
        num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        cate_cols = ['workclass', 'education', 'marital_status', 'relationship']
        label_col = 'income_bracket'
        data_arr_list, labels_arr_list, feature_names, labels_names = util.load_data_balance(UtilTest.data_file, num_cols, cate_cols, label_col=label_col)
        print("test_load_data_balance =================")
        print(len(data_arr_list))
        print('data_arr_list => %s' %str(data_arr_list[0].shape))
        print('labels_arr_list => %s' % str(labels_arr_list[0].shape))
        print(feature_names)
        print(labels_names)

if __name__ == '__main__':
    unittest.main()