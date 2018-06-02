from util import time_measure
from feature_selection.abstract.FeatureSelection import FeatureSelect
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from model.classifier import SVMClassifier
import numpy as np

class PLS(FeatureSelect):
    @time_measure
    def fit(self, data, labels=[], params={}):
        max_iter = params.get('max_iter', 100)
        n_components = params.get('n_components', None)

        if n_components is None:
            cv = params.get('cv', 10)
            kf = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
            mse_list = []
            for n_components in range(1, data.shape[1]+1):
                mse_cv = []
                for train_index, test_index in kf.split(data, labels):
                    X_train_, X_test_ = data[train_index], data[test_index]
                    y_train_, y_test_ = labels[train_index], labels[test_index]
                    model = PLSRegression(n_components=n_components,
                                          max_iter=max_iter,
                                          scale=False)
                    model.fit(X_train_, y_train_)
                    y_tr_predict_ = model.predict(X_train_)
                    y_te_predict_ = model.predict(X_test_)

                    clf = SVMClassifier()
                    clf.train(y_tr_predict_, y_train_)
                    y_predict_ = clf.predict(y_te_predict_)

                    mse = mean_squared_error(y_test_, y_predict_)
                    mse_cv.append(mse)

                mse_list.append(np.array(mse_cv).mean())

            n_components = mse_list.index(min(mse_list))+1

        print('selected n_components : ', n_components)

        self.model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=False)
        self.model.fit(data, labels)
        print(self.model.get_params()['n_components'])

        return self.model

    def transform(self, data):
        if self.model is None:
            print('need to fit ..... ')
            exit()
        return np.dot(data, self.model.x_rotations_)

    def get_n_components(self):
        if self.model is None:
            print('need to fit ..... ')
            exit()
        return self.model.get_params()['n_components']


