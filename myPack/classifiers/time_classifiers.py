import numpy as np
from keras import Input

from myPack.classifiers.classifier_layers import batch_layer, conv_layer, dense_layer, dropout_layer, \
    flatten_layer, pooling_layers
from myPack.classifiers.super_classifier import SUPERClassifier
from keras.layers import DepthwiseConv2D


class TimeClassifer(SUPERClassifier):
    def __init__(self, input_shape, output_shape, logits, save_path="", fig_path="", save_name="", batch_size=32,
                 epochs=300, patience=50, verbose=False, learning_rate=0.001):
        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        inp = Input(self._input_shape)

        x = conv_layer(inp, num_filters=(64, 64, 64), kernel_size=((1, 6), (1, 6), (1, 6)), use_2d=self._conv2d,
                       padding="same")
        x = pooling_layers(x, pool_size=(1, 2), stride=(1, 2), use_2d=self._conv2d)
        x = dropout_layer(x, rate=0.1, monte_carlo=monte_carlo)

        x = conv_layer(x, num_filters=(128, 128), kernel_size=((3, 3), (3, 3)), use_2d=self._conv2d, padding="same")
        x = pooling_layers(x, pool_size=(2, 2), stride=(2, 2), use_2d=self._conv2d)
        x = dropout_layer(x, rate=0.1, monte_carlo=monte_carlo)

        x = conv_layer(x, num_filters=(256, 256), kernel_size=((3, 3), (3, 3)), use_2d=self._conv2d, padding="same")
        x = pooling_layers(x, pool_size=(2, 2), stride=(2, 2), use_2d=self._conv2d)
        x = dropout_layer(x, rate=0.1, monte_carlo=monte_carlo)

        x = flatten_layer(x, use_flatten=True, use_2d=self._conv2d)
        x = dense_layer(x, neurons=(256, 32), apply_dropout=True, apply_batchnorm=True, dropout_rate=0.5,
                        monte_carlo=monte_carlo)

        model = self._finish_model(inp, x=x, output_shape=self._output_shape)

        return model


class ExperimentalClassifier(SUPERClassifier):
    def __init__(self, input_shape, output_shape, logits, save_path="", fig_path="", save_name="", batch_size=32,
                 epochs=300, patience=50, verbose=False, learning_rate=0.001):
        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        inp = Input(self._input_shape)

        x = conv_layer(inp, num_filters=(32, 10), kernel_size=((1, self.input_shape[0]), (1, 10)), use_2d=self._conv2d,
                       padding="same")
        x = pooling_layers(x, pool_size=(1, 4), stride=(1, 4), use_2d=self._conv2d)
        x = batch_layer(x)
        x = dropout_layer(x, rate=0.25, monte_carlo=monte_carlo)

        x = conv_layer(x, num_filters=(32, 32), kernel_size=((1, 8), (1, 4)), use_2d=self._conv2d,
                       padding="same")
        x = pooling_layers(x, pool_size=(1, 2), stride=(1, 2), use_2d=self._conv2d)
        x = batch_layer(x)
        x = dropout_layer(x, rate=0.25, monte_carlo=monte_carlo)

        x = conv_layer(x, num_filters=64, kernel_size=(3, 3), use_2d=self._conv2d, padding="same")
        x = pooling_layers(x, pool_size=(2, 2), stride=(2, 2), use_2d=self._conv2d, max_pooling=False)
        x = dropout_layer(x, rate=0.25, monte_carlo=monte_carlo)

        x = flatten_layer(x, use_flatten=True, use_2d=self._conv2d)
        x = dense_layer(x, neurons=32, apply_dropout=True, apply_batchnorm=True, dropout_rate=0.5,
                        monte_carlo=monte_carlo)

        model = self._finish_model(inp, x=x, output_shape=self._output_shape)

        return model


if __name__ == "__main__":

    a = ExperimentalClassifier(input_shape=(129, 1000), output_shape=1, logits=False)
    print(a.model.summary())