import numpy as np
import pandas as pd
import tensorflow as tf
from model.classifier.abstract.Classifier import Classifier
import hyperopt
import re

class RNNClassifier(Classifier):
    ACT_FUNC_DICT = {
        'relu': tf.nn.relu,
        # 'crelu': tf.nn.crelu,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu,
    }

    def train(self, data, labels=None, params={}):
        """

        :param data:
        :param labels:
        :param params:

        :return:
        """

        def rnn_model_fn(features, labels, mode, params):
            """Model function for CNN."""
            # Input Layer
            print('features["x"] ==>', features["x"].shape)
            input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1], 1])


            lstm_fw_cells = []
            lstm_bw_cells = []
            layers_params = params['hidden_layers']['layers']

            for idx, layer_params in enumerate(layers_params):
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=layer_params['num_units'])
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=layer_params['output_keep_prob'])
                lstm_fw_cells.append(lstm_fw_cell)

                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=layer_params['num_units'])
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                             output_keep_prob=layer_params['output_keep_prob'])
                lstm_bw_cells.append(lstm_bw_cell)

            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, input_layer, dtype=tf.float32)
            outputs = tf.contrib.layers.flatten(outputs)
            print('flat', outputs)

            dense_units = (int)(outputs.shape[1])/2
            outputs = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
            print('dense', outputs)
            outputs = tf.layers.dropout(inputs=outputs, rate=params['last_dense_layer_dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)
            print('dropout', outputs)
            # Logits layer
            logits = tf.layers.dense(inputs=outputs, units=params['n_classes'])
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
            model_fn=rnn_model_fn,
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
        num_units_list = np.linspace(10, 50, num=5).astype(np.int32).tolist()
        # activation_list = ['relu', 'elu', 'selu']
        dropout_rate_list = np.linspace(0.2, 0.8, num=13).tolist()
        return num_units_list, dropout_rate_list

    @staticmethod
    def get_default_space(num_epochs=4):
        num_units_list, dropout_rate_list = RNNClassifier._space()
        return {
            'n_classes': 2,
            'learning_rate': 0.001,
            'num_epochs': num_epochs,
            'hidden_layers': hyperopt.hp.choice('hidden_layers', [
                {
                    'n_layers': 1,
                    'layers': [
                        {
                            'num_units': hyperopt.hp.choice('num_units_1_1', num_units_list),
                            'output_keep_prob': hyperopt.hp.choice('output_keep_prob_1_1', dropout_rate_list),
                        },
                    ]
                }, {
                    'n_layers': 2,
                    'layers': [
                        {
                            'num_units': hyperopt.hp.choice('num_units_2_1', num_units_list),
                            'output_keep_prob': hyperopt.hp.choice('output_keep_prob_2_1', dropout_rate_list),
                        },{
                            'num_units': hyperopt.hp.choice('num_units_2_2', num_units_list),
                            'output_keep_prob': hyperopt.hp.choice('output_keep_prob_2_2', dropout_rate_list),
                        },

                    ]
                }
            ]),
            'last_dense_layer_dropout_rate': hyperopt.hp.choice('last_dense_layer_dropout_rate', dropout_rate_list),
        }

    @staticmethod
    def parsing_tune_result(best):
        num_units_list, dropout_rate_list = RNNClassifier._space()
        params = RNNClassifier.get_default_space()
        params['hidden_layers'] = {}
        if len(best) == 4:
            params['hidden_layers']['n_layers'] = 1
            params['hidden_layers']['layers'] = [{}]
        elif len(best) == 6:
            params['hidden_layers']['n_layers'] = 2
            params['hidden_layers']['layers'] = [{}]
        else:
            print('error')
            raise Exception

        p_units = re.compile('num_units_[0-9]_[0-9]')
        p_output_keep_prob_ = re.compile('output_keep_prob_[0-9]_[0-9]')
        for k in best.keys():
            if p_units.match(k):
                layer_idx = int(k.split('_')[3]) - 1
                params['hidden_layers']['layers'][layer_idx]['num_units'] = num_units_list[best[k]]
            elif p_output_keep_prob_.match(k):
                layer_idx = int(k.split('_')[4]) - 1
                params['hidden_layers']['layers'][layer_idx]['output_keep_prob'] = dropout_rate_list[best[k]]
            elif 'last_dense_layer_dropout_rate' == k:
                params['last_dense_layer_dropout_rate'] = dropout_rate_list[best[k]]
        return params


if __name__ == "__main__":
    pass
    # CNNClassifier()