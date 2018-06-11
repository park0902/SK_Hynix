import numpy as np
import tensorflow as tf
from model.abstract import Classifier
import hyperopt
import re

class DNNClassifier(Classifier):
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

        def dnn_model_fn(features, labels, mode, params):
            """Model function for CNN."""
            # Input Layer
            print('features["x"] ==>', features["x"].shape)
            net = tf.reshape(features["x"], [-1, features["x"].shape[1]])
            layers_params = params['hidden_layers']['layers']

            for idx, layer_params in enumerate(layers_params):
                if type(layer_params['activation']) == str:
                    layer_params['activation'] = DNNClassifier.ACT_FUNC_DICT[layer_params['activation']]
                net = tf.layers.dense(inputs=net, units=layer_params['n_units'], activation=layer_params['activation'])
                print('dense',  idx, net)
                net = tf.layers.dropout(inputs=net, rate=layer_params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)
                print('dropout', idx, net)

            # Logits layer
            logits = tf.layers.dense(inputs=net, units=params['n_classes'])
            print('logits', logits)
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
            model_fn=dnn_model_fn,
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
        n_units_list = np.linspace(25, 100, num=15).astype(np.int32).tolist()
        activation_list = ['relu', 'elu', 'selu']
        last_dense_layer_units_list = np.linspace(256, 1024, num=4).astype(np.int32).tolist()
        dropout_rate_list = np.linspace(0.1, 0.9, num=17).tolist()
        return n_units_list, activation_list, last_dense_layer_units_list, dropout_rate_list


    @staticmethod
    def get_default_space(num_epochs=4):
        n_units_list, activation_list, last_dense_layer_units_list, dropout_rate_list = DNNClassifier._space()
        return {
            'n_classes': 2,
            'learning_rate': 0.001,
            'num_epochs': num_epochs,
            'hidden_layers': hyperopt.hp.choice('hidden_layers', [
                {
                    'n_layers': 2,
                    'layers': [
                        {
                            'n_units': hyperopt.hp.choice('n_units_2_1', n_units_list),
                            'activation': hyperopt.hp.choice('activation_2_1', activation_list),
                            'dropout_rate': hyperopt.hp.choice('dropout_rate_2_1', dropout_rate_list),
                        },
                        {
                            'n_units': hyperopt.hp.choice('n_units_2_2', n_units_list),
                            'activation': hyperopt.hp.choice('activation_2_2', activation_list),
                            'dropout_rate': hyperopt.hp.choice('dropout_rate_2_2', dropout_rate_list),
                        },

                    ]
                },
                {
                    'n_layers': 3,
                    'layers': [
                        {
                            'n_units': hyperopt.hp.choice('n_units_3_1', n_units_list),
                            'activation': hyperopt.hp.choice('activation_3_1', activation_list),
                            'dropout_rate': hyperopt.hp.choice('dropout_rate_3_1', dropout_rate_list),
                        },
                        {
                            'n_units': hyperopt.hp.choice('n_units_3_2', n_units_list),
                            'activation': hyperopt.hp.choice('activation_3_2', activation_list),
                            'dropout_rate': hyperopt.hp.choice('dropout_rate_3_2', dropout_rate_list),
                        },
                        {
                            'n_units': hyperopt.hp.choice('n_units_3_3', n_units_list),
                            'activation': hyperopt.hp.choice('activation_3_3', activation_list),
                            'dropout_rate': hyperopt.hp.choice('dropout_rate_3_3', dropout_rate_list),
                        },
                    ]
                },
            ]),
            # 'last_dense_layer_units': hyperopt.hp.choice('last_dense_layer_units',last_dense_layer_units_list),
            # 'last_dense_layer_dropout_rate': hyperopt.hp.choice('last_dense_layer_dropout_rate',last_dense_layer_dropout_rate_list),


        }

    @staticmethod
    def parsing_tune_result(best):
        n_units_list, activation_list, last_dense_layer_units_list, dropout_rate_list = DNNClassifier._space()
        params = DNNClassifier.get_default_space()
        params['hidden_layers'] = {}
        if len(best) == 7:
            params['hidden_layers']['n_layers'] = 2
            params['hidden_layers']['layers'] = [{}, {}]
        elif len(best) == 10:
            params['hidden_layers']['n_layers'] = 3
            params['hidden_layers']['layers'] = [{}, {}, {}]
        else:
            print('error')
            raise Exception

        p_units = re.compile('n_units_[0-9]_[0-9]')
        p_activation = re.compile('activation_[0-9]_[0-9]')
        p_dropout = re.compile('dropout_rate_[0-9]_[0-9]')
        for k in best.keys():
            if p_units.match(k):
                layer_idx = int(k.split('_')[3])-1
                params['hidden_layers']['layers'][layer_idx]['n_units'] = n_units_list[best[k]]
            elif p_activation.match(k):
                layer_idx = int(k.split('_')[2])-1
                params['hidden_layers']['layers'][layer_idx]['activation'] = activation_list[best[k]]
            elif p_dropout.match(k):
                layer_idx = int(k.split('_')[3])-1
                params['hidden_layers']['layers'][layer_idx]['dropout_rate'] = dropout_rate_list[best[k]]

        return params
if __name__ == "__main__":
    pass
    # CNNClassifier()