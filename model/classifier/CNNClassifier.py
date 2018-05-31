import numpy as np
import pandas as pd
import tensorflow as tf
from model.classifier.abstract.Classifier import Classifier
import hyperopt

class CNNClassifier(Classifier):
    ACT_FUNC_DICT = {
        'relu': tf.nn.relu,
        # 'crelu': tf.nn.crelu,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu,
    }
    # def __init__(self):
    #     super(CNNClassifier, self).__init__()
    #     # self.data_df = None
    #     self.model = None

    # def load_data(self, file_path, num_cols, cate_cols, label_col='', **kwargs):
    #     """ 전처리함수. pandas 이용
    #     :param str file_path: csv 파일경로
    #     :param list num_cols: 숫자 column 이름 list
    #         >>> num_cols = ['age','education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    #     :param list cate_cols: 범주형 column 이름 list\
    #         >>> ['workclass', 'education', 'marital_status',  'relationship']
    #     :param str label_col: 라벨 column 이름
    #     :returns: 전처리된 학습 DataFrame, 숫자로 인코딩된 라벨 numpy 배열
    #     """
    #     # 파일 읽기
    #     self.data_df = pd.read_csv(file_path)
    #     num_df = self.data_df[num_cols]
    #
    #     cate_df = self.data_df[cate_cols]
    #     cate_df = pd.get_dummies(cate_df, columns=cate_cols)  # , dummy_na=True, sparse=True
    #
    #     new_df = pd.concat([num_df, cate_df], axis=1)
    #     new_df = new_df.fillna(0)
    #
    #     if label_col == '':
    #         return new_df.values.astype(np.float32)
    #     else:
    #         # labels = self._get_label(label_col)
    #         label_df = self.data_df[label_col]
    #         label_df = pd.get_dummies(label_df, columns=[label_col])  # , dummy_na=True, sparse=True
    #         labels = label_df.values.astype(np.int32)
    #         return new_df.values.astype(np.float32), labels

    # def get_train_params(self):
    #     return {
    #         'n_features': 0,
    #         'n_classes': 2,
    #         'conv_units': [
    #             {'filters': 32, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
    #             {'filters': 64, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
    #             {'filters': 32, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
    #         ],
    #         'pool_units': [
    #             {'pool_size': [2], 'strides': 2},
    #             {'pool_size': [2], 'strides': 2},
    #             {'pool_size': [10], 'strides': 10},
    #         ],
    #         'dense_units': 1024,
    #         'dropout_rate': 0.4
    #
    #     }

    def train(self, data, labels=None, params={}):
        """

        :param data:
        :param labels:
        :param params:

        :return:
        """

        def cnn_model_fn(features, labels, mode, params):
            """Model function for CNN."""
            # Input Layer
            # net = tf.reshape(features["x"], [-1, params['n_features'], 1])
            print('features["x"] ==>', features["x"].shape)
            net = tf.reshape(features["x"], [-1, features["x"].shape[1], 1])
            print('features["x"] ==>net, ', net)
            layers_params = params['hidden_layers']['layers']

            for idx, layer_params in enumerate(layers_params):
                if type(layer_params['activation']) == str:
                    layer_params['activation'] = CNNClassifier.ACT_FUNC_DICT[layer_params['activation']]
                # pool_unit = layer_params['pool_units'][idx]
                net = tf.layers.conv1d(inputs=net, filters=layer_params['filters'], kernel_size=layer_params['kernel_size'], padding=layer_params['padding'], activation=layer_params['activation'])
                print('conv', idx, net)
                net = tf.layers.max_pooling1d(inputs=net, pool_size=[layer_params['pool_size']], strides=layer_params['pool_size'])
                print('pool', idx, net)


            # flat = tf.reshape(net, [-1, layers_params[-1]['filters']])
            flat = tf.reshape(net, [-1, net.shape[1]* net.shape[2]])
            print('flat', flat)

            dense = tf.layers.dense(inputs=flat, units=params['last_dense_layer_units'], activation=tf.nn.relu)
            print('dense', dense)
            # Add dropout operation; 0.6 probability that element will be kept
            dropout = tf.layers.dropout(inputs=dense, rate=params['last_dense_layer_dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)
            print('dropout', dropout)
            # Logits layer
            logits = tf.layers.dense(inputs=dropout, units=params['n_classes'])
            print('logits', logits)
            # print('labels', labels.shape)
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Calculate Loss (for both TRAIN and EVAL modes)
            # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        self.model = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            # model_dir="/tmp/mnist_convnet_model"
            params=params
        )

        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # labels = labels.reshape(1, -1)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data}, y=labels,
            batch_size=100,
            num_epochs=params.get('num_epochs', 4),
            shuffle=True)

        return self.model.train(
            input_fn=train_input_fn,
            # steps=200,
            hooks=[logging_hook])

    def predict(self, data):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            batch_size=len(data), num_epochs=1, shuffle=False)
        results = self.model.predict(input_fn=input_fn)
        y_predict = [r['classes'] for r in results]
        return y_predict

    @staticmethod
    def _space():
        filters_list = np.linspace(2, 64, num=32).astype(np.int32).tolist()
        kernel_size_list = np.linspace(2, 6, num=3).astype(np.int32).tolist()
        padding_list = ['valid', 'same']
        # activation_list = ['relu', 'crelue', 'elu', 'selu']
        activation_list = ['relu', 'elu', 'selu']
        pool_size_list = np.linspace(3, 9 ,num=4).astype(np.int32).tolist()
        last_dense_layer_units_list = np.linspace(256, 1024, num=4).astype(np.int32).tolist()
        last_dense_layer_dropout_rate_list = np.linspace(0.1, 0.9, num=17).tolist()
        return filters_list, kernel_size_list, padding_list, activation_list, pool_size_list, \
                last_dense_layer_units_list, last_dense_layer_dropout_rate_list


    @staticmethod
    def get_default_space(num_epochs=4):
        filters_list, kernel_size_list, padding_list, activation_list, pool_size_list, last_dense_layer_units_list, last_dense_layer_dropout_rate_list = CNNClassifier._space()
        return {
            'n_classes': 2,
            'learning_rate': 0.001,
            'num_epochs': num_epochs,
            'hidden_layers': hyperopt.hp.choice('hidden_layers', [
                {
                    'n_layers': 2,
                    'layers': [
                        {
                            'filters': hyperopt.hp.choice('filters_2_1', filters_list),
                            'kernel_size': hyperopt.hp.choice('kernel_size_2_1', kernel_size_list),
                            'padding': hyperopt.hp.choice('padding_2_1', padding_list),
                            'activation': hyperopt.hp.choice('activation_2_1', activation_list),
                            'pool_size': hyperopt.hp.choice('pool_size_2_1', pool_size_list),
                        },
                        {
                            'filters': hyperopt.hp.choice('filters_2_2', filters_list),
                            'kernel_size': hyperopt.hp.choice('kernel_size_2_2', kernel_size_list),
                            'padding': hyperopt.hp.choice('padding_2_2', padding_list),
                            'activation': hyperopt.hp.choice('activation_2_2', activation_list),
                            'pool_size': hyperopt.hp.choice('pool_size_2_2', pool_size_list),
                        },
                    ]
                },
                {
                    'n_layers': 3,
                    'layers': [
                        {
                            'filters': hyperopt.hp.choice('filters_3_1', filters_list),
                            'kernel_size': hyperopt.hp.choice('kernel_size_3_1', kernel_size_list),
                            'padding': hyperopt.hp.choice('padding_3_1', padding_list),
                            'activation': hyperopt.hp.choice('activation_3_1', activation_list),
                            'pool_size': hyperopt.hp.choice('pool_size_3_1', pool_size_list),
                        },
                        {
                            'filters': hyperopt.hp.choice('filters_3_2', filters_list),
                            'kernel_size': hyperopt.hp.choice('kernel_size_3_2', kernel_size_list),
                            'padding': hyperopt.hp.choice('padding_3_2', padding_list),
                            'activation': hyperopt.hp.choice('activation_3_2', activation_list),
                            'pool_size': hyperopt.hp.choice('pool_size_3_2', pool_size_list),
                        },
                        {
                            'filters': hyperopt.hp.choice('filters_3_3', filters_list),
                            'kernel_size': hyperopt.hp.choice('kernel_size_3_3', kernel_size_list),
                            'padding': hyperopt.hp.choice('padding_3_3', padding_list),
                            'activation': hyperopt.hp.choice('activation_3_3', activation_list),
                            'pool_size': hyperopt.hp.choice('pool_size_3_3', pool_size_list),
                        },
                    ]
                },
            ]),
            'last_dense_layer_units': hyperopt.hp.choice('last_dense_layer_units',last_dense_layer_units_list),
            'last_dense_layer_dropout_rate': hyperopt.hp.choice('last_dense_layer_dropout_rate',last_dense_layer_dropout_rate_list),

            # 'conv_units': [
            #     {'filters': 32, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
            #     {'filters': 64, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
            #     {'filters': 32, 'kernel_size': 5, 'padding': 'same', 'activation': tf.nn.relu},
            # ],
            # 'pool_units': [
            #     {'pool_size': [2], 'strides': 2},
            #     {'pool_size': [2], 'strides': 2},
            #     {'pool_size': [10], 'strides': 10},
            # ],
            # 'dense_units': 1024,
            # 'dropout_rate': 0.4

        }

    @staticmethod
    def parsing_tune_result(best):
        filters_list, kernel_size_list, padding_list, activation_list, pool_size_list, last_dense_layer_units_list, last_dense_layer_dropout_rate_list = CNNClassifier._space()
        params = CNNClassifier.get_default_space()
        params['hidden_layers'] = {}
        if len(best) == 13:
            params['hidden_layers']['n_layers'] = 2
            params['hidden_layers']['layers'] = [{}, {}]
        elif len(best) == 18:
            params['hidden_layers']['n_layers'] = 3
            params['hidden_layers']['layers'] = [{}, {}, {}]
        else:
            print('error')
            raise Exception
        """
         {
            'filters': hyperopt.hp.choice('filters_3_1', filters_list),
            'kernel_size': hyperopt.hp.choice('kernel_size_3_1', kernel_list),
            'padding': hyperopt.hp.choice('padding_3_1', padding_list),
            'activation': hyperopt.hp.choice('activation_3_1', activation_list),
            'pool_size': hyperopt.hp.choice('pool_size_3_1', pool_size_list),
        },
        """

        for k in best.keys():
            if 'filters' in k:
                layer_idx = int(k.split('_')[2])-1
                params['hidden_layers']['layers'][layer_idx]['filters'] = filters_list[best[k]]
            elif 'kernel_size' in k:
                layer_idx = int(k.split('_')[3])-1
                params['hidden_layers']['layers'][layer_idx]['kernel_size'] = kernel_size_list[best[k]]
            elif 'padding' in k:
                layer_idx = int(k.split('_')[2])-1
                params['hidden_layers']['layers'][layer_idx]['padding'] = padding_list[best[k]]
            elif 'activation' in k:
                layer_idx = int(k.split('_')[2])-1
                params['hidden_layers']['layers'][layer_idx]['activation'] = activation_list[best[k]]
            elif 'pool_size' in k:
                layer_idx = int(k.split('_')[3])-1
                params['hidden_layers']['layers'][layer_idx]['pool_size'] = pool_size_list[best[k]]
            elif 'last_dense_layer_units' == k:
                params['last_dense_layer_units'] = last_dense_layer_units_list[best[k]]
            elif 'last_dense_layer_dropout_rate' == k:
                params['last_dense_layer_dropout_rate'] = last_dense_layer_dropout_rate_list[best[k]]

        return params
if __name__ == "__main__":
    pass
    # CNNClassifier()