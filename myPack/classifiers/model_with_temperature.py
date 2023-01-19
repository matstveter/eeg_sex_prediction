import sys

import keras
import numpy as np
import sklearn.calibration
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from matplotlib import pyplot as plt
import torch
from torch import nn, optim


class ModelWithTemperature:
    """
    Keras implementation of: https://github.com/gpleiss/temperature_scaling
    
    Thin decorator, which wraps a model with temperature scaling model.

    NB: Output of the model sent as input should be logits and NOT softmax or sigmoid
    """

    def __init__(self, model, batch_size, save_path, opt):
        self.model = model
        self.temperature = tf.compat.v1.get_variable(name="temperature", shape=(1,),
                                                     initializer=tf.constant_initializer(1.5))
        self.batch_size = batch_size
        self.opt = opt
        self.save_path = save_path + opt + "_reliability_"

    def temperature_scale(self, logits):
        temperature = tf.ones(logits.shape) * self.temperature
        return logits / temperature

    def set_temperature(self, valid_loader):
        nll_criterion = BinaryCrossentropy(from_logits=True)
        # ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        print("Loading data....")
        for input_val, label in valid_loader:
            logits = self.model(input_val)
            logits_list.append(logits)
            labels_list.append(label)
        logits = tf.concat(logits_list, axis=0)
        labels = tf.concat(labels_list, axis=0)
        print("Finished loading data!!")

        import tensorflow_probability as tfp
        self.calibration_reliability(logits=logits, labels=labels, name_ext="before")
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(y_true=labels, y_pred=logits)
        # before_temperature_ece = ece_criterion(logits, labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, 0))

        print(sklearn.metrics.brier_score_loss(y_true=labels, y_prob=tf.keras.activations.sigmoid(logits)))
        # Define the optimizer
        if self.opt == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate=0.01)
            print("Running SGD")
        elif self.opt == "adam":
            optimizer = tf.optimizers.Adam(learning_rate=0.01)
            print("Running Adam")
        elif self.opt == "ftrl":
            optimizer = tf.optimizers.Ftrl(learning_rate=0.01)
            print("Running FTRL")
        elif self.opt == "rms":
            optimizer = tf.optimizers.RMSprop(learning_rate=0.01)
            print("Running RMS")
        else:
            raise ValueError("Unrecognizable optimizer")

        # Define the evaluation function (closure)
        def closure():
            with tf.GradientTape() as tape:
                loss = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
                grads = tape.gradient(loss, self.temperature)
            optimizer.apply_gradients([(grads, self.temperature)])
            return loss

        # Take the optimization step
        temp_loss = sys.maxsize

        k = 0
        for i in range(5000):
            optimizer.minimize(closure, [self.temperature])

            if i % 10 == 0:
                l = closure()
                print("Loss at iteration: ", i, " : ", l)

                if l < temp_loss and l != 0.0:
                    temp_loss = l
                else:
                    print(f"Current loss: {l} -> Temp_loss: {temp_loss}")
                    k += 1
                    if k == 100:
                        break
        after_temperature_nll = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, 0))
        print(f'Optimal temperature scale: {self.temperature}')

        self.calibration_reliability(logits=self.temperature_scale(logits), labels=labels, name_ext="after")
        print(sklearn.metrics.brier_score_loss(y_true=labels, y_prob=tf.keras.activations.sigmoid(self.temperature_scale(logits))))

    def calibration_reliability(self, logits, labels, name_ext):
        # print(logits)
        logits = tf.keras.activations.sigmoid(logits)
        # print(logits)
        prob_true, prob_pred = sklearn.calibration.calibration_curve(y_true=labels, y_prob=logits)
        # print(prob_true)
        # print(prob_pred)
        disp = sklearn.calibration.CalibrationDisplay(prob_true=prob_true, prob_pred=prob_pred, y_prob=logits)
        disp.plot()
        plt.show()
        plt.savefig(self.save_path + name_ext)
        plt.close()


class ModelTemp:
    def __init__(self, model):
        super(ModelTemp, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        nll_criterion = BinaryCrossentropy(from_logits=True)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        print("Loading data....")
        for input_val, label in valid_loader:
            logits = self.model(input_val)
            logits_list.append(logits)
            labels_list.append(label)
        logits = tf.concat(logits_list, axis=0)
        labels = tf.concat(labels_list, axis=0)
        print("Finished loading data!!")

        # First: collect all the logits and labels for the validation set

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(y_true=labels, y_pred=logits)
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, 0))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(y_true=labels, y_pred=self.temperature_scale(logits))
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, 0))

        return self