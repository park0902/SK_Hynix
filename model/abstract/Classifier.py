import hyperopt
import time
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np
from model.abstract import Model


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()

    def evaluate(self, data, labels):
        """

        :param X_test:
        :param y_test:
            0 라벨: positive
            1 라벨: negative
        :return:
        """

        predict_lables = self.predict(data)
        TP = 0; FP = 0; TN = 0; FN = 0;
        for i, y_t in enumerate(predict_lables):
            y_p = predict_lables[i]
            if y_p == 0 and y_t==0 : # TP
                TP +=1
            elif y_p == 0 and y_t==1 : # FP
                FP += 1
            elif y_p == 1 and y_t==1 : # TN
                TN += 1
            elif y_p == 1 and y_t==0 : # FN
                FN += 1
        return self.binary_evaluate(TP, FP, TN, FN, labels, predict_lables, pos_label=0), y_predict

    @classmethod
    def _get_tune(cls, data, labels, cv=10):
        def tune(params):
            tmp_model = cls()
            start_time = time.time()
            status = hyperopt.STATUS_OK
            try:
                if cv <= 1:
                    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
                    tmp_model.train(X_train, labels=y_train, params=params)
                    eval_results, y_predict = tmp_model.evaluate(X_test, y_test)
                    loss = log_loss(y_test, y_predict)
                    print('loss', loss)
                else:
                    eval_list, y_predict_list = cls.cross_val(data, labels=labels, params=params, cv=cv)
                    loss = np.array([evl['loss'] for evl in evl_list]).mean()
                    print('loss ================================>', loss)
                    eval_results = {"TP" : np.rint(np.array([evl_['TP'] for evl in eval_list]).mean()}





                    evl['FP'] = np.rint(np.mean([evl['FP'] for evl in evl_list])),
                    evl['TN'] = np.rint(np.mean([evl['TN'] for evl in evl_list])),
                    evl['FN'] = np.rint(np.mean([evl['FN'] for evl in evl_list])),
                    evl['acc'] = np.mean([evl['acc'] for evl in evl_list]),
                    evl['err'] = np.mean([evl['err'] for evl in evl_list]),
                    evl['precision'] = np.mean([evl['precision'] for evl in evl_list]),
                    evl['recall'] = np.mean([evl['recall'] for evl in evl_list]),
                    evl['f1'] = np.mean([evl['f1'] for evl in evl_list])
                    }
                exe_time = time.time() - start_time
            except Exception as e:
                print('Error', e)
                loss = 1000000000000000000
                status = hyperopt.STATUS_FAIL
                eval_results = {}
                exe_time = 0
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
    def tuning(cls, data, labels,  space, max_evals, tune_algo='tpe', cv=10):
        tune = cls._get_tune(data, labels, cv=cv)
        trials = hyperopt.Trials()
        if tune_algo == 'tpe':
            algo = hyperopt.tpe.suggest
        elif tune_algo == 'random':
            algo = hyperopt.rand.suggest

        best = hyperopt.fmin(tune, space, algo=algo, max_evals=max_evals, trials=trials)
        return best, trials

    # @abstractmethod
    # def save_model(self, data):
    #     pass
    #
    # @abstractmethod
    # def load_model(self, data):
    #     pass

