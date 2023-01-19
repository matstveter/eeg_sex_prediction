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
                 patience=50, verbose=False, learning_rate=0.001):
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
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                                                      min_lr=0.00001)
        self._callbacks = (earlystop, csv_logger, mcp_save, reduce_lr)

        # todo Don't use both early-stopping and reduce_lr ?

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
            out = keras.layers.Dense(output_shape)(x)

        else:
            custom_loss = keras.losses.BinaryCrossentropy(from_logits=False)
            out = keras.layers.Dense(output_shape, activation="sigmoid")(x)

        model = keras.models.Model(inputs=input_to_model, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self._learning_rate, decay=0.0)
        model.compile(optimizer=optim, loss=custom_loss, metrics=self._metrics)
        return model

    def _eval_trained_models(self, val_x, val_y) -> None:
        """ During model.fit, the best performing models will be saved during training. This function is called after
        the model is trained, to find the best model between the model that has been saved during training, and the
        last model after training. Then the best models weights will be saved in the model folder.

        Args:
            val_x: validation set input
            val_y: validation set ground truth

        Returns:
            None
        """
        if val_y is not None:
            self._temp_model.load_weights(self._model_path + "weights.mdl.wrs.hdf5")
            _, y_predicted_temp = apply_sigmoid_probs_and_classify(self._temp_model.predict(x=val_x,
                                                                                            batch_size=self._batch_size),
                                                                   is_logits=self._logits)
            res_temp_model_acc = accuracy_score(y_true=val_y, y_pred=y_predicted_temp)

            _, y_predicted_model = apply_sigmoid_probs_and_classify(self._model.predict(x=val_x,
                                                                                        batch_size=self._batch_size),
                                                                    is_logits=self._logits)
            model_acc = accuracy_score(y_true=val_y, y_pred=y_predicted_model)

            if res_temp_model_acc >= model_acc:
                self._temp_model.save_weights(self._model_path + self._save_name + "_weights.hdf5")
            else:
                self._model.save_weights(self._model_path + self._save_name + "_weights.hdf5")
        else:
            self._temp_model.load_weights(self._model_path + "weights.mdl.wrs.hdf5")
            res_temp_model = self._temp_model.evaluate(x=val_x, batch_size=self._batch_size, verbose=False)
            acc_temp = res_temp_model[1]
            # auc_temp = res_temp_model[-1]

            res_model = self._model.evaluate(x=val_x, batch_size=self._batch_size, verbose=False)
            acc_model = res_model[1]
            # auc_model = res_model[-1]

            # todo Evaluate auc instead of acc???

            if acc_temp >= acc_model:
                self._temp_model.save_weights(self._model_path + self._save_name + "_weights.hdf5")
            else:
                self._model.save_weights(self._model_path + self._save_name + "_weights.hdf5")

    def fit(self, train_x, train_y, val_x, val_y, plot_test_acc=False, save_raw=False):
        if train_y is None and val_y is None:
            # This applies when train_x and val_x is on the type of generator
            history = self._model.fit(x=train_x, validation_data=val_x, epochs=self._epochs,
                                      callbacks=self._callbacks)
        else:
            history = self._model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=self._epochs,
                                      batch_size=self._batch_size, shuffle=True, callbacks=self._callbacks)

        # Evaluate the last model against the saved model during the training
        self._eval_trained_models(val_x=val_x, val_y=val_y)
        print("[INFO] Model evaluated and selected!")

        # Load the best model from the evaluation function
        print("[INFO] Loading MC and best model!")
        self._model.load_weights(self._model_path + self._save_name + "_weights.hdf5")
        self._mc_model.load_weights(self._model_path + self._save_name + "_weights.hdf5")

        if plot_test_acc:
            plot_accuracy(history=history, fig_path=self._fig_path, save_name=self._save_name)

        print("[INFO] Plotted Accuracy")

        if save_raw:
            save_to_pkl(history.history, path=self._fig_path, name=self._save_name + "_raw_files")

        print("[INFO] Saved raw")

        keras.backend.clear_session()

        return history.history

    def predict(self, x_test, y_test, return_metrics=False):
        y_pred = self._model.predict(x=x_test, batch_size=self._batch_size)

        y_sigmoid, y_predicted = apply_sigmoid_probs_and_classify(y_pred, is_logits=self._logits)
        if return_metrics:
            accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
            num_correct = accuracy_score(y_true=y_test, y_pred=y_predicted, normalize=False)
            report = classification_report(y_true=y_test, y_pred=y_predicted, labels=[0, 1],
                                           target_names=['male', 'female'], zero_division=0, output_dict=True)
            # print(self._model.evaluate(x=x_test, y=y_test, batch_size=self._batch_size))

            return accuracy, report, num_correct
        else:
            return y_pred, y_predicted

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
