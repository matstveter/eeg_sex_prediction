from keras import Input
from keras.constraints import max_norm
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, DepthwiseConv2D, Dropout, \
    Flatten, MaxPooling2D, SeparableConv2D, SpatialDropout2D
from tensorflow import keras
import tensorflow as tf
import keras.backend as K

from myPack.classifiers.classifier_layers import dense_layer
from myPack.classifiers.super_classifier import SUPERClassifier

assume_128 = 4
assume_256 = 2

add_to_kernel_length=False

class EEGnet(SUPERClassifier):
    # NB Assumes sampling rate of 128
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, dropout_rate=0.5, kern_length=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropout_type="Dropout", add_dense=None):

        self._dropout_rate = dropout_rate
        self._kernel_length = kern_length
        if add_to_kernel_length:
            self._kernel_length *= assume_128
        self._f1 = F1
        self._f2 = F2
        self._d = D
        self._norm_rate = norm_rate
        self._dropout_type = dropout_type
        self._dense_layer = add_dense

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):

        if self._dropout_type == "Dropout":
            dropoutType = Dropout
        elif self._dropout_type == "SpatialDropout2D":
            dropoutType = SpatialDropout2D
        else:
            raise ValueError("Dropout type must be of either Dropout or SpatialDropout2D")

        input1 = Input(shape=self._input_shape)

        ##################################################################
        block1 = Conv2D(self._f1, (1, self._kernel_length), padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self._input_shape[0], 1), use_bias=False,
                                 depth_multiplier=self._d,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(self._dropout_rate)(block1)

        block2 = SeparableConv2D(self._f2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(self._dropout_rate)(block2)

        flatten = Flatten(name='flatten')(block2)

        if self._dense_layer is not None:
            flatten = dense_layer(input_to_dense=flatten, neurons=self._dense_layer, apply_dropout=True,
                                  monte_carlo=monte_carlo, apply_batchnorm=True, dropout_rate=self._dropout_rate)

        if self._logits:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=True)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(self._norm_rate))(flatten)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(self._norm_rate),
                                     activation="sigmoid")(flatten)

        model = keras.models.Model(inputs=input1, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)

        return model


class EEGnet_SSVEP(SUPERClassifier):
    # NB Assumes sampling rate of 128
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, dropout_rate=0.5, kern_length=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropout_type="Dropout", add_dense=None):
        self._dropout_rate = dropout_rate
        self._kernel_length = kern_length

        if add_to_kernel_length:
            self._kernel_length *= assume_128

        self._f1 = F1
        self._f2 = F2
        self._d = D
        self._norm_rate = norm_rate
        self._dropout_type = dropout_type
        self._dense_layer = add_dense

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        if self._dropout_type == "Dropout":
            dropoutType = Dropout
        elif self._dropout_type == "SpatialDropout2D":
            dropoutType = SpatialDropout2D
        else:
            raise ValueError("Dropout type must be of either Dropout or SpatialDropout2D")

        input1 = Input(shape=self._input_shape)
        ##################################################################
        block1 = Conv2D(self._f1, (1, self._kernel_length), padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self._input_shape[0], 1), use_bias=False,
                                 depth_multiplier=self._d,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(self._dropout_rate)(block1)

        block2 = SeparableConv2D(self._f2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(self._dropout_rate)(block2)

        flatten = Flatten(name='flatten')(block2)

        if self._dense_layer is not None:
            flatten = dense_layer(input_to_dense=flatten, neurons=self._dense_layer, apply_dropout=True,
                                  monte_carlo=monte_carlo, apply_batchnorm=True, dropout_rate=self._dropout_rate)

        if self._logits:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=True)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(self._norm_rate))(flatten)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(self._norm_rate),
                                     activation="sigmoid")(flatten)

        model = keras.models.Model(inputs=input1, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)

        return model


class DeepConvNet(SUPERClassifier):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, dropout_rate=0.5, kern_length=10, add_dense=None):
        self._dropout_rate = dropout_rate
        self._dense_layer = add_dense
        self._kernel_length = kern_length

        if add_to_kernel_length:
            self._kernel_length *= assume_256

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    def _build_model(self, monte_carlo=False):
        input1 = Input(self._input_shape)
        block1 = Conv2D(25, (1, self._kernel_length), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input1)
        block1 = Conv2D(25, (self._input_shape[0], 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
        block1 = Dropout(self._dropout_rate)(block1)

        block2 = Conv2D(50, (1, self._kernel_length),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
        block2 = Dropout(self._dropout_rate)(block2)

        block3 = Conv2D(100, (1, self._kernel_length),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(self._dropout_rate)(block3)

        block4 = Conv2D(200, (1, self._kernel_length),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(self._dropout_rate)(block4)

        flatten = Flatten()(block4)

        if self._dense_layer is not None:
            flatten = dense_layer(input_to_dense=flatten, neurons=self._dense_layer, apply_dropout=True,
                                  monte_carlo=monte_carlo, apply_batchnorm=True, dropout_rate=self._dropout_rate)

        if self._logits:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=True)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(0.5))(flatten)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(0.5),
                                     activation="sigmoid")(flatten)

        model = keras.models.Model(inputs=input1, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)

        return model


class ShallowNet(SUPERClassifier):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, dropout_rate=0.5, add_dense=None):
        self._dropout_rate = dropout_rate
        self._dense_layer = add_dense
        self._kernel_length = 25

        if add_to_kernel_length:
            self._kernel_length *= assume_256

        super().__init__(input_shape=input_shape, output_shape=output_shape, save_path=save_path, fig_path=fig_path,
                         save_name=save_name, logits=logits, batch_size=batch_size, epochs=epochs,
                         patience=patience, verbose=verbose, learning_rate=learning_rate)

    @staticmethod
    def square(x):
        return K.square(x)

    @staticmethod
    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))

    def _build_model(self, monte_carlo=False):
        input1 = Input(self._input_shape)
        block1 = Conv2D(40, (1, self._kernel_length), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input1)
        block1 = Conv2D(40, (self._input_shape[0], 1), use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(self.square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
        block1 = Activation(self.log)(block1)
        block1 = Dropout(self._dropout_rate)(block1)
        flatten = Flatten()(block1)

        if self._dense_layer is not None:
            flatten = dense_layer(input_to_dense=flatten, neurons=self._dense_layer, apply_dropout=True,
                                  monte_carlo=monte_carlo, apply_batchnorm=True, dropout_rate=self._dropout_rate)

        if self._logits:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=True)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(0.5))(flatten)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(self._output_shape, kernel_constraint=max_norm(0.5),
                                     activation="sigmoid")(flatten)

        model = keras.models.Model(inputs=input1, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)

        return model
