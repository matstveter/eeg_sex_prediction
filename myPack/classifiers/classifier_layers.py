from typing import Union
import keras
import keras.losses
from keras.layers import AveragePooling1D, AveragePooling2D, BatchNormalization, Concatenate, Conv1D, Conv2D, Dense, \
    Dropout, Flatten, GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D
import tensorflow as tf


def conv_layer(input_to_conv: tuple, num_filters: Union[int, tuple], kernel_size: Union[int, tuple], use_2d: bool,
               padding: str = "valid", kernel_init: str = "glorot_uniform", name: str = None, use_activation=True,
               data_format="channels_first"):
    """ Function which creates n number of conv layers, n is based on either that filters is a tuple containing
    filters like (32, 32, 32) -> 3 * conv layers with kernel_size as filter. Same thing applies for kernel_size.
    If both kernel_size and filters are tuples of length more than 1, they have to be similar length, and each layer
    will have kernel_size and filters based on the tuple

    Args:
        input_to_conv: input data, which will be sent through the conv layers
        num_filters: int if only one filter should be used, on one layer or alle layers(based on length of kernel size, or
                 tuple if the filters should be different from conv layer to conv layer
        kernel_size: int or tuple(len(1)) if only one kernel_size should be used on one or n layers based on number of
                     filters, or tuple if the kernels_size should be different across the different layers
        use_2d: whether to use 2D or 1D convolution
        padding: "same" if zero padding
        kernel_init: standard "glorot_uniform", see keras-> kernel initializers
        name: name of a conv layer, if needed for algorithms such as gradcam, however, this argument does not apply, if
              either kernel_size or filters is longer than one
        use_activation: Apply "relu" to the conv layers or not
        data_format: when 1D conv, specify this

    Raises:
        TypeError: filters or kernel_size is another type than [int, tuple], or if a tuple is sent as kernel size to
                   1D Conv
        ValueError: if some tuples are 0 in length
        assertError: if both filters and kernel_size is tuples of length >1, and they are not the same length

    Returns:
        output from the last layer
    """
    # If kernel size is an int, or kernel size is a tuple of length one, means that either it is only one layer with
    # this kernel size, or it is based on the length of filter. if filters is a int, only one layer is returned, else
    # the number of filters in the input tuple is returned
    if use_activation:
        activation = "relu"
    else:
        activation = None

    if isinstance(kernel_size, tuple) and len(kernel_size) > 0 and not use_2d:
        if isinstance(kernel_size[0], tuple):
            temp_list = []
            for k in kernel_size:
                temp_list.append(max(k))
            kernel_size = tuple(temp_list)
            print("[INFO] Convolutional Layer 1D does not work when Kernel Size is a tuple, "
                  f"max value of the tuple is taken as the new kernel_size! New kernel: {kernel_size}")

    if isinstance(kernel_size, int) or (isinstance(kernel_size, tuple) and len(kernel_size) == 1):
        if isinstance(num_filters, int):
            if use_2d:
                outp = Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation=activation,
                              kernel_initializer=kernel_init, name=name)(input_to_conv)
            else:
                outp = Conv1D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation=activation,
                              kernel_initializer=kernel_init, name=name,
                              data_format=data_format)(input_to_conv)
        elif isinstance(num_filters, tuple):
            if len(num_filters) == 1:
                if use_2d:
                    outp = Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation=activation,
                                  kernel_initializer=kernel_init, name=name)(input_to_conv)
                else:
                    outp = Conv1D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation=activation,
                                  kernel_initializer=kernel_init, name=name,
                                  data_format=data_format)(input_to_conv)
            elif len(num_filters) > 1:
                x = input_to_conv
                if use_2d:
                    for f in num_filters:
                        x = Conv2D(filters=f, kernel_size=kernel_size, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init)(x)
                else:
                    for f in num_filters:
                        x = Conv1D(filters=f, kernel_size=kernel_size, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init, data_format=data_format)(x)
                outp = x
            else:
                raise ValueError("Length of filters (tuple) must be more than 0")
        else:
            raise TypeError(f"Unrecognized type for filters, expected int or tuple, got {type(num_filters)}")
    # If kernel_size is a tuple, create len(tuple) layers, unless it is 1, then the if statement is True, or 0 which
    # will raise a ValueError.
    # If both kernel_size and filters are tuples and longer than 1, assert that they are the same length
    elif isinstance(kernel_size, tuple):
        if len(kernel_size) > 1:
            x = input_to_conv
            if isinstance(num_filters, int) or (isinstance(num_filters, tuple) and len(num_filters) == 1):
                if use_2d:
                    for k in kernel_size:
                        x = Conv2D(filters=num_filters, kernel_size=k, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init)(x)
                else:
                    for k in kernel_size:
                        x = Conv1D(filters=num_filters, kernel_size=k, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init, data_format=data_format)(x)
                outp = x
            else:
                assert len(num_filters) == len(kernel_size), "Mismatch between number of kernels and filters"
                if use_2d:
                    for f, k in zip(num_filters, kernel_size):
                        x = Conv2D(filters=f, kernel_size=k, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init)(x)
                else:
                    for f, k in zip(num_filters, kernel_size):
                        x = Conv1D(filters=f, kernel_size=k, padding=padding, activation=activation,
                                   kernel_initializer=kernel_init, data_format=data_format)(x)
                outp = x
        else:
            raise ValueError("Length of kernel_size must be more than 0")
    else:
        raise TypeError(f"Unrecognized type of kernel_size input argument, should be tuple: {type(kernel_size)}")
    return outp


def dense_layer(input_to_dense: tuple, neurons: Union[int, tuple], kernel_init, apply_batchnorm: bool = False,
                apply_dropout: bool = True, dropout_rate: float = 0.5, monte_carlo: bool = False):
    """ Function creating 1 or n numbers of dense-layers, based on neurons' argument, if tuple, len(tuple) dense layers
    is returned. This function can also apply dropout inbetween the layers, and also capable of creating it monte_carlo
    dropout ready. Relu is always applied, after batch_norm if True, and before dropout if true-

    Args:
        input_to_dense: input to be sent through the layers
        neurons: num neurons, either int or len(tuple) = 1 for single layer, or len(tuple) num layers
        apply_batchnorm: apply batch norm after dense layer
        apply_dropout: apply dropout
        dropout_rate: dropout rate
        monte_carlo: argument to the dropout layer function, set training=True on the dropout layers
        kernel_init: initializer for the weights, standard=glorot_uniform

    Raises:
        ValueError: If the number of neurons sent into the network is 0
        ValueError: If the input type is other than ints or tuples

    Returns:
        output from the dense layer
    """
    if not apply_dropout and monte_carlo:
        print("[INFO] Monte Carlo cant be set to True, while apply dropout is not, changing the apply_dropout to True!")
        apply_dropout = True

    # If neurons is a single int or a tuple of length 1, return 1 dense layer, with possible batch_norm and dropout
    if isinstance(neurons, int) or (isinstance(neurons, tuple) and len(neurons) == 1):
        x = Dense(neurons, kernel_initializer=kernel_init)(input_to_dense)
        if apply_batchnorm:
            x = BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if apply_dropout:
            x = dropout_layer(x, rate=dropout_rate, monte_carlo=monte_carlo)
        outp = x
    elif isinstance(neurons, tuple):
        if len(neurons) > 1:
            x = input_to_dense
            for n in neurons:
                x = Dense(n)(x)

                if apply_batchnorm:
                    x = BatchNormalization()(x)
                x = keras.layers.ReLU()(x)

                if apply_dropout:
                    x = dropout_layer(x, rate=dropout_rate, monte_carlo=monte_carlo)
            outp = x
        else:
            raise ValueError(f"Length of neurons tuple must be more than 0, got {len(neurons)}")
    else:
        raise ValueError(f"Type is not recognized, should be int or tuple, got: {type(neurons)}")
    return outp


def pooling_layers(input_to_pool: tuple, pool_size: Union[int, tuple], stride: Union[int, tuple], use_2d: bool,
                   max_pooling=True):
    """ Function that creates a pooling layer, maxpooling is standard, but can be changed to average pooling, by setting
    max_pooling to false

    Args:
        input_to_pool: input to pool layer
        pool_size: int or tuple setting the size of the pooling kernel
        stride: stride of the kernel
        max_pooling: Default = True, if False -> Averagepooling
        use_2d: Pooling operations on 2D or 1D

    Returns:
        output from maxpooling layer
    """
    if not use_2d and isinstance(pool_size, tuple):
        pool_size = max(pool_size)
        if isinstance(stride, tuple):
            stride = max(stride)
        print("[INFO] MaxPool1D does not work when pool_size is a tuple, extracting the max value from the pool size as"
              f"the new pool size and the same for stride! New size: {pool_size} with stride {stride}")

    if max_pooling:
        if use_2d:
            outp = MaxPooling2D(pool_size=pool_size, strides=stride)(input_to_pool)
        else:
            outp = MaxPooling1D(pool_size=pool_size, strides=stride, data_format="channels_first")(input_to_pool)
    else:
        if use_2d:
            outp = AveragePooling2D(pool_size=pool_size, strides=stride)(input_to_pool)
        else:
            outp = AveragePooling1D(pool_size=pool_size, strides=stride, data_format="channels_first")(input_to_pool)
    return outp


def dropout_layer(input_to_drop: tuple, rate: float, monte_carlo: bool = False):
    """ Function to create dropout layer. Can apply monte carlo dropout, meaning that training=True, during inference

    Args:
        input_to_drop: input to layers
        rate: dropout rate
        monte_carlo: training=True during inference

    Raises:
        ValueError: If the dropout rate is 0 or above 1.

    Returns:
        output from dropout layer
    """
    if rate == 0.0 or rate > 1.0:
        raise ValueError(f"Dropout rate should be between 0.0 and 1.0, got {rate}")

    if monte_carlo:
        outp = Dropout(rate=rate)(input_to_drop, training=True)
    else:
        outp = Dropout(rate=rate)(input_to_drop)
    return outp


def flatten_layer(input_to_flatten: tuple, use_2d, use_flatten: bool = True):
    """ Function that can either flatten a layer or use globalaveragepooling

    Args:
        input_to_flatten: input to layer
        use_flatten: flatten or not
        use_2d: if not flatten, average and flatte, if not use globalaveragepooling1d

    Returns:
        output from flatten layer
    """
    if use_flatten:
        outp = Flatten()(input_to_flatten)
    else:
        if use_2d:
            # Instead of GlobalAveragePooling
            input_to_flatten = tf.reduce_mean(input_to_flatten, axis=2)
            outp = Flatten()(input_to_flatten)

        else:
            outp = GlobalAveragePooling1D(data_format="channels_first")(input_to_flatten)
    return outp


def inception_module(input_to_inception, kernel_init, activation: str = "linear", use_bottleneck: bool = False,
                     bottleneck_size: int = 32, kernel_sizes: tuple = None, num_filters: int = 32):
    """ Function to create a inception module, taken from
    https://github.com/hfawaz/InceptionTime/blob/470ce144c1ba43b421e72e1d216105db272e513f/classifiers/inception.py

    Args:
        kernel_init: weight initializer
        input_to_inception: input to module
        activation: activation function used throughout the module, standard=Linear
        use_bottleneck: use bottlenect layers or not
        bottleneck_size: How many filters for the bottleneck layer
        kernel_sizes: 3 or more kernels used for applying on the input tensor if None, it is 10, 20 and 40
        num_filters: how many filters per conv layers

    Returns:
        output from module
    """
    if len(input_to_inception.shape) > 3:
        raise TypeError("Trying to use 2D data in inception time, will not work!")

    if use_bottleneck and int(input_to_inception.shape[-1] > 1):
        input_tensor = Conv1D(filters=bottleneck_size, kernel_size=1, padding="same", activation=activation,
                              use_bias=False, data_format="channels_first",
                              kernel_initializer=kernel_init)(input_to_inception)
    else:
        input_tensor = input_to_inception

    if kernel_sizes is None:
        kernel_sizes = (10, 20, 40)

    conv_list = []

    for k in kernel_sizes:
        conv_list.append(Conv1D(filters=num_filters, kernel_size=k, strides=1, padding="same", activation=activation,
                                use_bias=False, data_format="channels_first",
                                kernel_initializer=kernel_init)(input_tensor))

    max_pool_1 = MaxPooling1D(pool_size=3, strides=1, padding="same")(input_tensor)
    conv_lay = Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=activation,
                      use_bias=False, data_format="channels_first", kernel_initializer=kernel_init)(max_pool_1)
    conv_list.append(conv_lay)

    x = Concatenate(axis=1)(conv_list)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def shortcut_layer(input_tensor, output_tensor, kernel_init="glorot_uniform"):
    shortcut_y = Conv1D(filters=int(output_tensor.shape[1]), kernel_size=1, padding="same", use_bias=False,
                        data_format="channels_first",
                        kernel_initializer=kernel_init)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, output_tensor])
    x = keras.layers.ReLU()(x)
    return x


def batch_layer(input_to_batch):
    """ Function to add a batch normalization_layer

    Args:
        input_to_batch:

    Returns:
        Output from batchnorm
    """
    return BatchNormalization()(input_to_batch)
