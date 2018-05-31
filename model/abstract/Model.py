from abc import *
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy import stats
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm

import pickle
import os

from sklearn import preprocessing, svm

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

    @staticmethod
    def binary_evaluate(TP, FP, TN, FN):
        eval_result ={ 'TP': TP, 'FP':FP, 'TN':TN, 'FN':FN}
        if (TP + FP + TN + FN) != 0:
            eval_result['acc'] = (TP + TN) / (TP + FP + TN + FN)
            eval_result['err'] = (FP + FN) / (TP + FP + TN + FN)
        if (TP + FP) != 0:
            eval_result['precision'] = TP / (TP + FP)
        if (TP + FN) != 0:
            eval_result['recall'] = TP / (TP + FN)
        if (eval_result['precision'] + eval_result['recall'] ) != 0:
            eval_result['f1'] = 2 * eval_result['precision'] * eval_result['recall'] / (eval_result['precision'] + eval_result['recall'] )
        return eval_result

    # @abstractmethod
    # def evaluate(self, data, labels):
    #     pass

    @classmethod
    def cross_val(cls, data, labels=[], params={}, cv=10): # SVDD 는 안되겄다.
        kf = KFold(n_splits=cv)
        kf.get_n_splits(data)
        evl_list =[]
        y_predict_list = []
        for train_index, test_index in kf.split(data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model = cls()
            # if len(labels) == 0: # svdd
            #     model.train(X_train, params=params)
            # else:
            #     model.train(X_train, y_train, params=params)
            model.train(X_train, y_train, params=params)
            evl, y_predict = model.evaluate(X_test, y_test)
            evl['loss'] = log_loss(y_test, y_predict)
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


