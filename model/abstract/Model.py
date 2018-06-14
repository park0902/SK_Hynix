import time
from abc import *
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


class Model(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def train(self, data, labels=None, params={}, **kwargs):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)

    @staticmethod
    def binary_clf_evaluate(TP, FP, TN, FN, y_true, y_pred, pos_label=0):
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
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
            result['auc'] = metrics.auc(fpr, tpr)
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
        kf = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
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

            eval_result, y_predict = model.evaluate(X_test, y_test)
            eval_result['loss'] = metrics.log_loss(y_test, y_predict)
            eval_result['exe_time'] = time.time() - start_time
            evl_list.append(eval_result)
            y_predict_list.append(y_predict)
        print('evl_list ================>', evl_list)
        return evl_list, y_predict_list

