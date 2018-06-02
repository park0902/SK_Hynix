from util import time_measure
from feature_selection.abstract.FeatureSelection import FeatureSelect
import numpy as np
import math
from sklearn.linear_model import LassoCV

class Lasso(FeatureSelect):
    @time_measure
    def fit(self, data, labels=[], params={}):
        cv = params.get('cv', 10)
        max_iter = params.get('mat_iter', 100)
        n_alphas = params.get('n_alphas', 500)
        option = params.get('option', 'mse+std')

        self.model = LassoCV(n_alphas=n_alphas,
                             max_iter=max_iter,
                             cv=cv,
                             fit_intercept=False,
                             normalize=False,
                             random_state=1)
        self.model.fit(data, labels)

        if option == 'mse+std': # (교차검증 MSE 최소값 + 교차검증 MSE 표준편차)의 MSE값과 가장 가까운 alpha 값 사용
            mean_mse_list = [mse_cv.mean() for mse_cv in self.model.mse_path_]
            std_mse_list = [mse_cv.std() for mse_cv in self.model.mse_path_]

            min_mean_mes = min(mean_mse_list)   # 교차검증 MSE 최소값
            min_mean_mes_std = std_mse_list[mean_mse_list.index(min(mean_mse_list))]    # 교차검증 MSE 표준편차

            obj_val = min_mean_mes + min_mean_mes_std / math.sqrt(cv)

            alpha_idx_mean_mse = np.abs(mean_mse_list - min_mean_mes).argmin()
            alpha_idx_obj_val = np.abs(mean_mse_list - (obj_val)).argmin()
            is_positive = (alpha_idx_obj_val - alpha_idx_mean_mse) > 0
            alpha_idx = alpha_idx_mean_mse

            gap_mse = 999999999
            while True:
                gap = np.abs(mean_mse_list[alpha_idx] - mean_mse_list[alpha_idx_obj_val])
                if gap > gap_mse:
                    if is_positive:
                        alpha_idx -= 1
                    else:
                        alpha_idx += 1
                    break
                else:
                    gap_mse = gap

                if is_positive:
                    alpha_idx += 1
                    if alpha_idx == len(mean_mse_list):
                        alpha_idx = len(mean_mse_list) - 1
                        break
                else:
                    alpha_idx -= 1
                    if alpha_idx < 0:
                        alpha_idx = 0
                        break
            print('alpha_idx_mean_mse : ', alpha_idx_mean_mse)
            print('alpha_idx_obj_val : ', alpha_idx_obj_val)
            print('alpha_idx : ', alpha_idx)

            self.model = LassoCV(alphas=[self.model.alpha_[alpha_idx]],
                                 max_iter=max_iter,
                                 cv=cv,
                                 fit_intercept=False,
                                 normalize=False)
            self.model.fit(data, labels)

        elif option == 'mse':
            pass
        return self.model

    def transform(self, data):
        if self.model is None:
            print('need to fit ..... ')
            exit()
        selected_idx_list = self.get_selected_idx_list()
        return self.model.coef_[selected_idx_list]*data[:,selected_idx_list]


    def get_selected_idx_list(self):
        if self.model is None:
            print('need to fit ..... ')
            exit()
            coef_idx_list = []
            for i, c in enumerate(self.model.coef_):
                if c != 0:
                    coef_idx_list.append(i)
            return coef_idx_list

    def get_n_components(self):
        if self.model is None:
            print('need to fit ..... ')
            exit()
        return len(self.get_selected_idx_list())