import sys

import numpy as np
import sklearn.calibration
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, minimize


class ModelWithTemperature:
    """
    Keras implementation of: https://github.com/gpleiss/temperature_scaling
    
    Thin decorator, which wraps a model with temperature scaling model.

    NB: Output of the model sent as input should be logits and NOT softmax or sigmoid
    """

    def __init__(self, model, batch_size, save_path):
        self.model = model
        self.temperature = tf.compat.v1.get_variable(name="temperature", shape=(1,),
                                                     initializer=tf.constant_initializer(1.5))
        self.batch_size = batch_size
        self.save_path = save_path + "_reliability_"

    def temperature_scale(self, logits):
        temperature = tf.ones(logits.shape) * self.temperature
        return logits / temperature

    def set_temperature(self, valid_loader):
        nll_criterion = BinaryCrossentropy(from_logits=True)
        # ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        for input_val, label in valid_loader:
            logits = self.model(input_val)
            logits_list.append(logits)
            labels_list.append(label)
        logits = tf.concat(logits_list, axis=0)
        labels = tf.concat(labels_list, axis=0)

        self.calibration_reliability(logits=logits, labels=labels, name_ext="before")
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(y_true=labels, y_pred=logits)
        # before_temperature_ece = ece_criterion(logits, labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, 0))

        # Define the optimizer
        optimizer = tf.optimizers.SGD(learning_rate=0.01)

        # Define the evaluation function (closure)
        def closure():
            with tf.GradientTape() as tape:
                loss = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
                grads = tape.gradient(loss, self.temperature)
            optimizer.apply_gradients([(grads, self.temperature)])
            return loss

        # Take the optimization step
        temp_loss = sys.maxsize
        print(temp_loss)

        for i in range(25000):
            optimizer.minimize(closure, [self.temperature])

            if i % 10 == 0:
                l = closure()
                print("Loss at iteration: ", i, " : ", l)

                if l < temp_loss or l == 0.0:
                    temp_loss = l
                else:
                    print(f"Current loss: {l} -> Temp_loss: {temp_loss}")
                    break

        after_temperature_nll = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, 0))
        print(f'Optimal temperature scale: {self.temperature}')

        self.calibration_reliability(logits=self.temperature_scale(logits), labels=labels, name_ext="after")

    def calibration_reliability(self, logits, labels, name_ext):
        # print(logits)
        logits = tf.keras.activations.sigmoid(logits)
        # print(logits)
        prob_true, prob_pred = sklearn.calibration.calibration_curve(y_true=labels, y_prob=logits)
        # print(prob_true)
        # print(prob_pred)
        disp = sklearn.calibration.CalibrationDisplay(prob_true=prob_true, prob_pred=prob_pred, y_prob=logits)
        disp.plot()
        plt.savefig(self.save_path + name_ext)
