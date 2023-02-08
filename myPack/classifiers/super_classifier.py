import abc
import keras.backend
import numpy as np
import tensorflow as tf
import keras
import keras.losses
import keras.layers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

from myPack.classifiers.keras_utils import mcc, specificity, recall, precision, f1, apply_sigmoid_probs_and_classify
from myPack.utils import plot_accuracy, save_to_pkl


class SUPERClassifier(abc.ABC):
    def __init__(self, input_shape, output_shape, save_path, fig_path, save_name, logits, batch_size=32, epochs=300,
                 patience=50, verbose=False, learning_rate=0.001, kernel_init="glorot_uniform"):
        self._input_shape = input_shape

        if len(input_shape) == 3:
            self._conv2d = True
        elif len(input_shape) == 2:
            self._conv2d = False
        else:
            raise ValueError(f"Wrong input_shape, expected 3, or 4 dim, got {len(input_shape)}")

        self._logits = logits
        self._output_shape = output_shape

        # Training Hyperparameters
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self._learning_rate = learning_rate
        self._kernel_init = kernel_init

        # Different saving paths
        self._model_path = save_path
        self._fig_path = fig_path
        self._save_name = save_name

        # Metrics
        self._metrics = ('accuracy', mcc, specificity, recall, f1, precision,
                         tf.keras.metrics.AUC(from_logits=self._logits))

        # Callbacks
        self._monitor = "val_loss"  # Could also be val_loss
        earlystop = EarlyStopping(monitor=self._monitor, patience=patience, verbose=True, mode="auto")
        csv_logger = CSVLogger(self._model_path + "training.csv", append=True)

        # Standard is val_loss
        mcp_save = ModelCheckpoint(self._model_path + "weights.mdl.wrs.hdf5", save_best_only=True, verbose=0)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                                      min_lr=0.00001)
        self._callbacks = (earlystop, csv_logger, mcp_save, reduce_lr)

        # Two models, self.model will be the last model after training, early stop model is the one with the suggested
        # best weights during training
        self._model = self._build_model()
        self._temp_model = self._build_model()
        self._mc_model = self._build_model(monte_carlo=True)

        # Print Model architect
        if verbose:
            self._model.summary()

    # ----------------#
    # Properties      #
    # ----------------#
    @property
    def save_name(self):
        return self._save_name

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def logits(self):
        return self._logits

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def model(self):
        return self._model

    @property
    def mc_model(self):
        return self._mc_model

    # ----------------#
    # End Properties  #
    # ----------------#

    def _finish_model(self, input_to_model, x, output_shape):
        if self._logits:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=True)
            out = keras.layers.Dense(output_shape, kernel_initializer=self._kernel_init)(x)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(output_shape, activation="sigmoid", kernel_initializer=self._kernel_init)(x)

        model = keras.models.Model(inputs=input_to_model, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)
        return model

    def _eval_trained_models(self, validation_generator) -> None:
        """ During model.fit, the best performing models will be saved during training. This function is called after
        the model is trained, to find the best model between the model that has been saved during training, and the
        last model after training. Then the best models weights will be saved in the model folder.

        Args:
            validation_generator: validation set generator

        Returns:
            None
        """
        self._temp_model.load_weights(self._model_path + "weights.mdl.wrs.hdf5")
        res_temp_model = self._temp_model.evaluate(x=validation_generator, batch_size=self._batch_size, verbose=False)
        acc_temp = res_temp_model[1]
        # auc_temp = res_temp_model[-1]

        res_model = self._model.evaluate(x=validation_generator, batch_size=self._batch_size, verbose=False)
        acc_model = res_model[1]
        # auc_model = res_model[-1]

        # todo Evaluate auc instead of acc???

        if acc_temp >= acc_model:
            self._temp_model.save_weights(self._model_path + self._save_name + "_weights.hdf5")
        else:
            self._model.save_weights(self._model_path + self._save_name + "_weights.hdf5")

    def fit(self, train_generator, validation_generator, plot_test_acc=False, save_raw=False):
        history = self._model.fit(x=train_generator, validation_data=validation_generator, epochs=self._epochs,
                                  callbacks=self._callbacks)

        # Evaluate the last model against the saved model during the training
        self._eval_trained_models(validation_generator=validation_generator)

        # Load the best model from the evaluation function
        self._model.load_weights(self._model_path + self._save_name + "_weights.hdf5")
        self._mc_model.load_weights(self._model_path + self._save_name + "_weights.hdf5")

        if plot_test_acc:
            plot_accuracy(history=history, fig_path=self._fig_path, save_name=self._save_name)

        if save_raw:
            save_to_pkl(history.history, path=self._fig_path, name=self._save_name + "_raw_files")

        keras.backend.clear_session()
        return history

    def predict(self, data, labels=None, return_metrics=False, verbose=False, monte_carlo=False):
        """ Predicts input data, can be either a data generator or a testx/testy dataset

        Args:
            data: dataGenerator or numpy array
            labels:
            return_metrics:
            verbose:
            monte_carlo

        Returns:

        """
        if return_metrics:
            if isinstance(data, np.ndarray) and labels is None:
                raise ValueError("Need labels as well if the data argument is not a data-generator")

            # Returns evaluation metrics
            if monte_carlo:
                eval_metrics = self._mc_model.evaluate(x=data, y=labels, batch_size=self._batch_size,
                                                       return_dict=True, verbose=verbose)
            else:
                eval_metrics = self._model.evaluate(x=data, y=labels, batch_size=self._batch_size, return_dict=True,
                                                    verbose=verbose)
            return eval_metrics
        else:
            # Predicts, sends through sigmoid if the logits is True and then classifies
            if monte_carlo:
                y_pred = self._mc_model.predict(x=data, batch_size=self._batch_size, verbose=verbose)
            else:
                y_pred = self._model.predict(x=data, batch_size=self._batch_size, verbose=verbose)
            y_sigmoid, y_classes = apply_sigmoid_probs_and_classify(y_prediction=y_pred, is_logits=self._logits)
            return y_pred, y_sigmoid, y_classes

    def __repr__(self):
        return f"Model Name: {self._save_name}"

    @abc.abstractmethod
    def _build_model(self, monte_carlo=False):
        """ Creates the keras architect of an entire model, and returns the model

        Args:
            monte_carlo: if monte carlo dropout should be applied or not

        Returns:
            a built keras model
        """
