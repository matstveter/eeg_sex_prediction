import numpy as np
from keras import Input
from keras.constraints import max_norm
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, DepthwiseConv2D, Dropout, \
    Flatten, SeparableConv2D, SpatialDropout2D
from tensorflow import keras
import tensorflow as tf

from myPack.classifiers.super_classifier import SUPERClassifier


class EEGnet(SUPERClassifier):
    # NB Assumes sampling rate of 128
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, dropout_rate=0.5, kern_length=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropout_type="Dropout"):

        self._dropout_rate = dropout_rate
        self._kernel_length = kern_length
        self._f1 = F1
        self._f2 = F2
        self._d = D
        self._norm_rate = norm_rate
        self._dropout_type = dropout_type

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
                 norm_rate=0.25, dropout_type="Dropout"):
        self._dropout_rate = dropout_rate
        self._kernel_length = kern_length
        self._f1 = F1
        self._f2 = F2
        self._d = D
        self._norm_rate = norm_rate
        self._dropout_type = dropout_type

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


if __name__ == "__main__":
    # a = EEGnet(input_shape=(129, 1000, 1), output_shape=1, logits=False, save_path="", fig_path="", save_name="")
    # print(a.model.summary())
    #
    # x = np.zeros((32, 129, 1000, 1))
    # y = np.zeros(32)
    # x1 = np.zeros((32, 129, 1000, 1))
    # y1 = np.zeros(32)
    #
    # a.fit(train_x=x, train_y=y, val_x=x1, val_y=y1)

    a = EEGnet_SSVEP(input_shape=(129, 1000, 1), output_shape=1, logits=False, save_path="", fig_path="", save_name="")
    print(a.model.summary())

    x = np.zeros((32, 129, 1000, 1))
    y = np.zeros(32)
    x1 = np.zeros((32, 129, 1000, 1))
    y1 = np.zeros(32)

    a.fit(train_x=x, train_y=y, val_x=x1, val_y=y1)
