import numpy as np
import sklearn.tree
# import sklearn as sk
import hyperopt

from model.abstract import Classifier


class DTClassifier(Classifier):
    # def __init__(self):
    #     super(SVMClassifier, self).__init__()
    #     self.data_df = None
    #     self.clf = None

    def train(self, data, labels=[], params={}):
        """

        :param data:
        :param labels:
        :param params:

        :return:
        """
        criterion = params.get('criterion ', 'gini')
        splitter = params.get('splitter ', 'best')
        max_depth = params.get('max_depth ', None)
        min_samples_leaf = params.get('min_samples_leaf ', 1)
        max_features = params.get('max_features ', None)
        self.model = sklearn.tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        return self.model.fit(data, labels)

    def save_image(self, out_file, feature_names=[], class_name=[]):
        if feature_names == []:
            feature_names = None
        if class_name == []:
            class_name = None
        dot = sklearn.tree.export_graphviz(
            self.model, out_file,
            feature_names=feature_names, class_name=class_name,
            filled=True, rounded=True, special_characters=True)
        print("현재 경로에서 아래 명령어를 실행")
        print("dot -Tpng %s -o %.png" %(out_file, out_file))

    @staticmethod
    def _space():
        criterion_list = ['gini', 'entropy']
        splitter_list = ['best', 'random']
        max_depth_list = np.linspace(5, 50, num=10).astype(np.int32)
        min_samples_leaf = np.linspace(1, 10, num=10).astype(np.int32)
        max_features_list = [None, 'sqrt', 'log2']
        return criterion_list, splitter_list, max_depth_list, min_samples_leaf, max_features_list

    @staticmethod
    def get_default_space(max_iter=200):
        criterion_list, splitter_list, max_depth_list, min_samples_leaf, max_features_list = DTClassifier._space()
        return {
            'criterion': hyperopt.hp.choice('criterion', criterion_list),
            'splitter': hyperopt.hp.choice('splitter', splitter_list),
            'max_depth': hyperopt.hp.choice('max_depth', max_depth_list),
            'min_samples_leaf': hyperopt.hp.choice('min_samples_leaf', min_samples_leaf),
            'max_features': hyperopt.hp.choice('max_features', max_features_list),
        }

    @staticmethod
    def parsing_tune_result(best):
        criterion_list, splitter_list, max_depth_list, min_samples_leaf, max_features_list = DTClassifier._space()
        params = {}
        for k in best.keys():
            if 'criterion' == k:
                params['criterion'] = criterion_list[best[k]]
            elif 'splitter' == k:
                params['splitter'] = splitter_list[best[k]]
            elif 'max_depth' == k:
                params['max_depth'] = max_depth_list[best[k]]
            elif 'min_samples_leaf' == k:
                params['min_samples_leaf'] = min_samples_leaf[best[k]]
            elif 'max_features' == k:
                params['max_features'] = max_features_list[best[k]]
        return params

