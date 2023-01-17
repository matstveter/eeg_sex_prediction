import keras
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
import tensorflow as tf


def recall(y_true, y_pred):
    """Calculates the recall (true positive rate) for a set of true labels and predicted labels.

       Args:
           y_true: A tensor or array of true labels with shape (n_samples,)
           y_pred: A tensor or array of predicted labels with shape (n_samples,)

       Returns:
           A float value representing the recall (true positive rate).
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    """Calculates the precision for a set of true labels and predicted labels.

        Args:
            y_true: A tensor or array of true labels with shape (n_samples,)
            y_pred: A tensor or array of predicted labels with shape (n_samples,)

        Returns:
            A float value representing the precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())

    return precision_keras


def specificity(y_true, y_pred):
    """Calculates the specificity (true negative rate) for a set of true labels and predicted labels.

        Args:
            y_true: A tensor or array of true labels with shape (n_samples,)
            y_pred: A tensor or array of predicted labels with shape (n_samples,)

        Returns:
            A float value representing the specificity (true negative rate).
    """
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def f1(y_true, y_pred):
    """Calculates the F1 score for a set of true labels and predicted labels.

        The F1 score is the harmonic mean of precision and recall.

        Args:
            y_true: A tensor or array of true labels with shape (n_samples,)
            y_pred: A tensor or array of predicted labels with shape (n_samples,)

        Returns:
            A float value representing the F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def mcc(y_true, y_pred):
    """Calculates the Matthews correlation coefficient (MCC) for a set of true labels and predicted labels.

        Args:
            y_true: A tensor or array of true labels with shape (n_samples,)
            y_pred: A tensor or array of predicted labels with shape (n_samples,)

        Returns:
            A float value representing the Matthews correlation coefficient.
        """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def get_classification_accuracy_from_array(sample_list: list, label: int):
    """ Classifies a list of samples, using majority voting, and returns the accuracy and num_correct samples

    Args:
        sample_list: list of samples, time_windows
        label: either 0 for male or 1 for female

    Returns:
        float: accuracy of the given samples'
        int: number of correctly classified samples
    """
    if len(sample_list) == 0:
        accuracy = 0
        num_samples = 0
    else:
        true_labels = np.array([label] * len(sample_list))
        accuracy = accuracy_score(y_true=true_labels, y_pred=np.array(sample_list))
        num_samples = accuracy_score(y_true=true_labels, y_pred=np.array(sample_list), normalize=False)

    return accuracy, num_samples


def apply_sigmoid_probs_and_classify(y_prediction, is_logits: bool):
    """

    Args:
        y_prediction: predicted
        is_logits: if the predictions are logits or the output from sigmoid
        data_y when calculating probs

    Returns:

    """
    if is_logits:
        y_pred = np.copy(y_prediction)
        y_sigmoid = tf.keras.activations.sigmoid(y_pred)
    else:
        y_sigmoid = np.copy(y_prediction)
    y_classified = keras.backend.round(y_sigmoid).numpy()

    return y_sigmoid, y_classified

