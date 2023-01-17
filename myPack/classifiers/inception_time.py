import keras.layers
from keras import Input

from myPack.classifiers.classifier_layers import dense_layer, inception_module, shortcut_layer
from myPack.classifiers.super_classifier import SUPERClassifier


class InceptionTime(SUPERClassifier):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, bottleneck=True, bottleneck_size=32,
                 use_residual=True, depth=6, nb_filters=32):

        self._bottleneck = bottleneck
        self._bottleneck_size = bottleneck_size
        self._use_residual = use_residual
        self._depth = depth
        self._nb_filters = nb_filters

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        input_layer = Input(self._input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self._depth):
            x = inception_module(input_to_inception=x,
                                 use_bottleneck=self._bottleneck,
                                 bottleneck_size=self._bottleneck_size,
                                 kernel_sizes=(10, 20, 40),
                                 num_filters=self._nb_filters)

            if self._use_residual and d % 3 == 2:
                x = shortcut_layer(input_tensor=input_res, output_tensor=x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        model = self._finish_model(input_to_model=input_layer, x=gap_layer, output_shape=self._output_shape)
        return model


class InceptionTime2(SUPERClassifier):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, bottleneck=True, bottleneck_size=32,
                 use_residual=True, depth=6, nb_filters=32, kernel_sizes=(10, 20, 40)):

        self._kernel_sizes = kernel_sizes
        self._bottleneck = bottleneck
        self._bottleneck_size = bottleneck_size
        self._use_residual = use_residual
        self._depth = depth
        self._nb_filters = nb_filters

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        input_layer = Input(self._input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self._depth):
            x = inception_module(input_to_inception=x,
                                 use_bottleneck=self._bottleneck,
                                 bottleneck_size=self._bottleneck_size,
                                 kernel_sizes=self._kernel_sizes,
                                 num_filters=self._nb_filters)

            if self._use_residual and d % 3 == 2:
                x = shortcut_layer(input_tensor=input_res, output_tensor=x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        gap_layer = dense_layer(input_to_dense=gap_layer, neurons=32, apply_dropout=True, apply_batchnorm=True,
                                monte_carlo=monte_carlo)
        model = self._finish_model(input_to_model=input_layer, x=gap_layer, output_shape=self._output_shape)
        return model


class InceptionTime3(SUPERClassifier):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, bottleneck=True, bottleneck_size=32,
                 use_residual=True, depth=6, nb_filters=32, kernel_sizes=(10, 20, 40)):

        self._kernel_sizes = kernel_sizes
        self._bottleneck = bottleneck
        self._bottleneck_size = bottleneck_size
        self._use_residual = use_residual
        self._depth = depth
        self._nb_filters = nb_filters

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        input_layer = Input(self._input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self._depth):
            x = inception_module(input_to_inception=x,
                                 use_bottleneck=self._bottleneck,
                                 bottleneck_size=self._bottleneck_size,
                                 kernel_sizes=self._kernel_sizes,
                                 num_filters=self._nb_filters)

            if self._use_residual and d % 3 == 2:
                x = shortcut_layer(input_tensor=input_res, output_tensor=x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        gap_layer = dense_layer(input_to_dense=gap_layer, neurons=(64, 16), apply_dropout=True, apply_batchnorm=True,
                                monte_carlo=monte_carlo)
        model = self._finish_model(input_to_model=input_layer, x=gap_layer, output_shape=self._output_shape)
        return model
