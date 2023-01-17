import keras.backend
import numpy as np
from myPack.classifiers.keras_utils import get_classification_accuracy_from_array


class CategorySampleUncertainty:
    __slots__ = "_mean_predictions", "_predictive_entropy", "_label", "_category_threshold", "_ent_thresholds", \
        "_attention_prediction_correct"

    def __init__(self, mean_prediction: np.ndarray, predictive_entropy: np.ndarray, true_label: int):
        if mean_prediction.shape[0] != predictive_entropy.shape[0]:
            print(f"[ERROR] Mean predictions and predictive entropy has mismatching shapes: {mean_prediction.shape} : "
                  f"{predictive_entropy.shape}")
            raise ValueError("CategorySampleUncertainty class: init method, mismatching shapes!")
        self._mean_predictions = mean_prediction
        self._predictive_entropy = predictive_entropy
        self._label = true_label
        self._category_threshold = [np.array([0.95, 0.05]), np.array([0.9, 0.1]), np.array([0.8, 0.2]),
                                    np.array([0.7, 0.3]), np.array([0.6, 0.4])]
        self._ent_thresholds = [pred_entropy(thresh, scale=False) for thresh in self._category_threshold]

        # Values used later
        self._attention_prediction_correct = 0

    # --------------
    # Properties
    # --------------
    @property
    def ent_thresholds(self):
        return self._ent_thresholds

    @property
    def attention_prediction_correct(self):
        return self._attention_prediction_correct

    # --------------
    # End Properties
    # --------------

    def calculate_category_results(self):
        """ Loops through the thresholds from the entropy and extracts the predictions where the entropy falls below
        the threshold, and then evaluates the kept and remaining samples.
        """
        kept_samples_per_threshold = list()
        num_cor_kept_per_threshold = list()
        num_cor_rej_per_threshold = list()

        for entropy_thresholds in self._ent_thresholds:
            temp_list_kept = list()
            temp_list_rejected = list()

            for ent, pred in zip(self._predictive_entropy, self._mean_predictions):
                if ent < entropy_thresholds:
                    if pred > 0.5:
                        # if the mean of predictions is above 0.5, the class is 1
                        temp_list_kept.append(1)
                    else:
                        temp_list_kept.append(0)
                else:
                    if pred > 0.5:
                        # if the mean of predictions is above 0.5, the class is 1
                        temp_list_rejected.append(1)
                    else:
                        temp_list_rejected.append(0)

            _, num_kept = get_classification_accuracy_from_array(sample_list=temp_list_kept, label=self._label)
            _, num_rej = get_classification_accuracy_from_array(sample_list=temp_list_rejected, label=self._label)

            num_cor_kept_per_threshold.append(num_kept)
            num_cor_rej_per_threshold.append(num_rej)
            kept_samples_per_threshold.append(len(temp_list_kept))

        return kept_samples_per_threshold, num_cor_kept_per_threshold, num_cor_rej_per_threshold

    def get_single_category(self, threshold, scale=False):
        """ Evaluates the predictions based on the predictive entropy and a threshold for keeping or rejecting a
        sample, if the predictive entropy is too high

        Args:
            threshold:
            scale: Scale the predictive entropy to **2
        Returns:

        """
        if type(threshold) == list:
            threshold = pred_entropy(np.array(threshold), scale=scale)

        if scale:
            pred_ent = self._predictive_entropy**2
        else:
            pred_ent = self._predictive_entropy

        kept_sample = list()
        rejected_sample = list()

        for ent, pred in zip(pred_ent, self._mean_predictions):
            if ent < threshold:
                if pred > 0.5:
                    kept_sample.append(1)
                else:
                    kept_sample.append(0)
            else:
                if pred > 0.5:
                    rejected_sample.append(1)
                else:
                    rejected_sample.append(0)

        _, num_kept_cor = get_classification_accuracy_from_array(sample_list=kept_sample, label=self._label)
        _, num_rej_cor = get_classification_accuracy_from_array(sample_list=rejected_sample, label=self._label)

        return len(kept_sample), num_kept_cor, len(rejected_sample), num_rej_cor

    def attention_vector(self, scale=False):
        """ Predicts by creating an attention vector from the uncertainty measurements, and scales the mean predictions
        with the uncertainty, effectively meaning that the samples with low uncertainty contributes more to the final
        prediction

        Args:
            scale: scale the predictive entropy by factor of 2(**2) or not

        Returns:

        """
        if scale:
            pred_ent = self._predictive_entropy**2
        else:
            pred_ent = self._predictive_entropy

        certainty = (1-pred_ent)
        certain_scaled_vector = certainty / np.sum(certainty, axis=0)
        majority_predictions = np.squeeze(keras.backend.round(self._mean_predictions).numpy(), axis=1)

        attention_pred = int(keras.backend.round(np.dot(a=majority_predictions, b=certain_scaled_vector)).numpy())

        if attention_pred == self._label:
            self._attention_prediction_correct = 1


def pred_entropy(probs: np.ndarray, scale=True) -> float:
    """ Calculated the predictive entropy from either a list of [1, 2] vectors or just a [1,2] vector
    Args:
        scale: Whether to scale it by **2
        probs: numpy array with shape [1, 2] or more [N, 2]

    Returns:
        float: predictive entropy
    """
    if scale:
        return ((-np.sum(probs * np.log(probs + 1e-12), axis=-1)) / np.log(2)) ** 2
    else:
        return (-np.sum(probs * np.log(probs + 1e-12), axis=-1)) / np.log(2)


def get_mutual_information_and_variance(y_sigmoid):
    total_var = np.var(y_sigmoid, axis=0)
    return total_var


def calculate_uncertainty_metrics(predictions: list, output_from_sigmoid: list):
    """ Functon for calculating uncertainty metrics, mean, variance, and predictive entropy
    Args:
        output_from_sigmoid:
        predictions: Predictions output from ensembles
    """
    predictions = np.array(predictions)
    output_from_sigmoid = np.array(output_from_sigmoid)

    # Used for predictive entropy
    y_prob_class1 = np.copy(output_from_sigmoid)
    y_prob_class0 = 1 - y_prob_class1
    prob0_mean = np.mean(y_prob_class0, axis=0)
    prob1_mean = np.mean(y_prob_class1, axis=0)
    class_probabilities = np.column_stack((prob0_mean, prob1_mean))

    predictive_entropy = pred_entropy(class_probabilities, scale=False)

    get_mutual_information_and_variance(y_sigmoid=output_from_sigmoid)
    return predictive_entropy, np.mean(predictions, axis=0)
