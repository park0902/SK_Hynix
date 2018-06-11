import time
from abc import *

import sklearn.externals.joblib
import sklearn.metrics
import sklearn.model_selection


# from sklearn.model_selection import KFold
# from sklearn.metrics import log_loss
# from scipy import stats
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing, svm
#
# import pickle
# import os
#
# from sklearn import preprocessing, svm

class Model(metaclass=ABCMeta):

    # @staticmethod
    # def _get_label_encoder(encoder_path):
    #     if os.path.exists(encoder_path):
    #         with open(encoder_path, 'rb') as f:
    #             return pickle.load(f)
    #     else:
    #         return None
    #
    # @staticmethod
    # def _make_label_encoder(encoder_path, labels_value):
    #     encoder = preprocessing.LabelEncoder()
    #     encoder.fit(labels_value)
    #     with open(encoder_path, 'wb') as f:
    #         pickle.dump(encoder, f)
    #     return encoder
    #
    # def _get_label(self, label_col):
    #     label_df = self.data_df[label_col]
    #     encoder_path = './col_%s.pkl'
    #     encoder = self._get_label_encoder(encoder_path)
    #     if encoder is None:
    #         encoder = self._make_label_encoder(encoder_path, label_df.values)
    #     encoder.fit(label_df.values)
    #     labels = encoder.transform(label_df.values)
    #     return labels
    #
    # @abstractmethod
    # def load_data(self, file_path, num_cols, cate_cols, label_col='', **kwargs):
    #
    #     pass

    @abstractmethod
    def train(self, data, labels=None, params={}):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self, file_path):
        sklearn.externals.joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = sklearn.externals.joblib.load(file_path)

    @staticmethod
    def binary_evaluate(TP, FP, TN, FN, y_true, y_pred, pos_label=0):
        """

        :param TP:
        :param FP:
        :param TN:
        :param FN:
        :param y_true:
        :param y_pred:
        :param pos_label: positive-label
        :return:
        """
        result ={'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
        try:
            fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
            result['auc'] = sklearn.metrics.auc(fpr, tpr)
            result['acc'] = (TP + TN) / (TP + FP + TN + FN)
            result['err'] = (FP + FN) / (TP + FP + TN + FN)

            if (TP + FP) != 0:
                result['precision'] = TP / (TP + FP)
            else:
                result['precision'] = 0

            if (TP + FN) != 0:
                result['recall'] = TP / (TP + FN)
            else:
                result['recall'] = 0

            if (result['precision'] + result['recall'] ) != 0:
                result['f1'] = 2 * result['precision'] * result['recall'] / (result['precision'] + result['recall'] )
            else:
                result['f1'] = 0
        except Exception as e:
            print('Error => ', e)

        return result

    # @abstractmethod
    # def evaluate(self, data, labels):
    #     pass

    @classmethod
    def cross_val(cls, data, labels=[], params={}, cv=10, **kwargs): # SVDD 는 안되겄다.
        kf = sklearn.model_selection.StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
        evl_list =[]
        y_predict_list = []
        for train_index, test_index in kf.split(data, labels):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            start_time = time.time()
            model = cls()

            outlier_val = kwargs.get('outlier_val', None)
            if outlier_val is not None:
                model.train(X_train, y_train, params=params, outlier_val=outlier_val)
            else:
                model.train(X_train, y_train, params=params)

            evl, y_predict = model.evaluate(X_test, y_test)
            evl['loss'] = sklearn.metrics.log_loss(y_test, y_predict)
            evl['exe_time'] = time.time() - start_time
            evl_list.append(evl)
            y_predict_list.append(y_predict)
        print('evl_list ================>', evl_list)
        return evl_list, y_predict_list


    # @abstractmethod
    # def save_model(self, data):
    #     pass
    #
    # @abstractmethod
    # def load_model(self, data):
    #     pass


