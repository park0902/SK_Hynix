import numpy as np
import sklearn.svm
# import sklearn as sk
import hyperopt

from model.abstract.Classifier import Classifier


class SVMClassifier(Classifier):
    # def __init__(self):
    #     super(SVMClassifier, self).__init__()
    #     self.data_df = None
    #     self.clf = None

    def train(self, data, labels=[], params={}):
        """

        :param data:
        :param labels:
        :param params:

        C : float, optional (default=1.0)
            Penalty parameter C of the error term.
        kernel : string, optional (default=’rbf’)
            ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
        :return:
        """
        max_iter = params.get('max_iter', 100)
        params = params.get('type', {})
        C = params.get('C', 1.0)
        kernel = params.get('kernel', 'rbf')
        degree = params.get('degree', 3)
        gamma = params.get('gamma', 'auto')
        self.model = sklearn.svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter)
        return self.model.fit(data, labels)

    @staticmethod
    def _space():
        C_list = np.logspace(-5, 9, num=15, base=2)
        gamma_list = np.logspace(-15, 3, num=19, base=2)
        degree_list = np.linspace(2, 15, num=14)
        # kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
        kernel_list = ['rbf']
        return C_list, gamma_list, degree_list, kernel_list

    @staticmethod
    def get_default_space(max_iter=200):
        C_list, gamma_list, degree_list, kernel_list = SVMClassifier._space()
        return {
            'max_iter': max_iter,
            'type': hyperopt.hp.choice('type', [
                # {'C': hyperopt.hp.choice('C_1', C_list), 'kernel': kernel_list[0]},
                # {'C': hyperopt.hp.choice('C_2', C_list), 'kernel': kernel_list[1], 'gamma': hyperopt.hp.choice('gamma_2', gamma_list)},
                # {'C': hyperopt.hp.choice('C_3', C_list), 'kernel': kernel_list[2], 'gamma': hyperopt.hp.choice('gamma_3', gamma_list), 'degree': hyperopt.hp.choice('degree', degree_list)},
                # {'C': hyperopt.hp.choice('C_4', C_list), 'kernel': kernel_list[3], 'gamma': hyperopt.hp.choice('gamma_4', gamma_list)},

                {'C': hyperopt.hp.choice('C_2', C_list), 'kernel': kernel_list[0],'gamma': hyperopt.hp.choice('gamma_2', gamma_list)},
            ])
        }

    @staticmethod
    def parsing_tune_result(best):
        C_list, gamma_list, degree_list, kernel_list = SVMClassifier._space()
        params = SVMClassifier.get_default_space()
        params['type'] = {}
        for k in best.keys():
            if 'C' in k:
                params['type']['C'] = C_list[best[k]]
            elif 'type' in k:
                params['type']['kernel'] = kernel_list[best[k]]
            elif 'gamma' in k:
                params['type']['gamma'] = gamma_list[best[k]]
            elif 'degree' in k:
                params['type']['degree'] = degree_list[best[k]]
        return params

