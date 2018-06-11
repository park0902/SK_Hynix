import hyperopt
# from scipy import stats
# from sklearn.metrics import log_loss
import numpy as np
# from model.abstract.Model import Model
from model.abstract import Model
# import model.abstract.Model


# import pandas as pd
# from sklearn import preprocessing, svm

# from sklearn import preprocessing, svm

class Classifier(Model):

    def evaluate(self, X_test, y_test):
        """

        :param X_test:
        :param y_test:
            0 라벨: positive
            1 라벨: negative
        :return:
        """

        y_predict = self.predict(X_test)
        TP = 0; FP = 0; TN = 0; FN = 0;
        for i, y_t in enumerate(y_test):
            y_p = y_predict[i]
            if y_p == 0 and y_t==0 : # TP
                TP +=1
            elif y_p == 0 and y_t==1 : # FP
                FP += 1
            elif y_p == 1 and y_t==1 : # TN
                TN += 1
            elif y_p == 1 and y_t==0 : # FN
                FN += 1
        return self.binary_evaluate(TP, FP, TN, FN, y_test, y_predict, pos_label=0), y_predict

    @classmethod
    def _get_tune(cls, data, labels, params={}, cv=10):
        def tune(params):
            status = hyperopt.STATUS_OK
            try:
                evl_list, y_predict_list = cls.cross_val(data, labels=labels, params=params, cv=cv)
                loss = np.mean([evl['loss'] for evl in evl_list])
                print('loss ================================>', loss)
                evl = dict()
                evl['TP'] = np.rint(np.mean([evl['TP'] for evl in evl_list]))
                evl['FP'] = np.rint(np.mean([evl['FP'] for evl in evl_list]))
                evl['TN'] = np.rint(np.mean([evl['TN'] for evl in evl_list]))
                evl['FN'] = np.rint(np.mean([evl['FN'] for evl in evl_list]))
                evl['acc'] = np.mean([evl['acc'] for evl in evl_list])
                evl['err'] = np.mean([evl['err'] for evl in evl_list])
                evl['precision'] = np.mean([evl['precision'] for evl in evl_list])
                evl['recall'] = np.mean([evl['recall'] for evl in evl_list])
                evl['f1'] = np.mean([evl['f1'] for evl in evl_list])
            except Exception as e:
                print('Error', e)
                loss = 10000000
                status = hyperopt.STATUS_FAIL
                evl = {}
            finally:
                return {'loss': loss, 'status': status, 'eval': evl}
            # evl_list, y_predict_list = cls.cross_val(data, labels=labels, params=params, cv=cv)
            # loss = np.mean([evl['loss'] for evl in evl_list])
            # print('loss ================================>', loss)
            # evl = dict()
            # evl['TP'] = np.rint(np.mean([evl['TP'] for evl in evl_list]))
            # evl['FP'] = np.rint(np.mean([evl['FP'] for evl in evl_list]))
            # evl['TN'] = np.rint(np.mean([evl['TN'] for evl in evl_list]))
            # evl['FN'] = np.rint(np.mean([evl['FN'] for evl in evl_list]))
            # evl['acc'] = np.mean([evl['acc'] for evl in evl_list])
            # evl['err'] = np.mean([evl['err'] for evl in evl_list])
            # evl['precision'] = np.mean([evl['precision'] for evl in evl_list])
            # evl['recall'] = np.mean([evl['recall'] for evl in evl_list])
            # evl['f1'] = np.mean([evl['f1'] for evl in evl_list])
            # return {'loss': loss, 'status': status, 'eval': evl}
        return tune


    @classmethod
    def tuning(cls, data, labels,  space, max_evals, params={}, mode='tpe', cv=10):
        if mode == 'tpe':
            algo = hyperopt.tpe.suggest
        elif mode == 'random':
            algo = hyperopt.rand.suggest
        tune = cls._get_tune(data, labels, params=params, cv=cv)
        trials = hyperopt.Trials()
        best = hyperopt.fmin(tune, space, algo=algo, max_evals=max_evals, trials=trials)
        return best, trials

    # @abstractmethod
    # def save_model(self, data):
    #     pass
    #
    # @abstractmethod
    # def load_model(self, data):
    #     pass

