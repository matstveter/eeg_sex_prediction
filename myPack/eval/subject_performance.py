import numpy as np
from sklearn.metrics import accuracy_score
from myPack.classifiers.keras_utils import apply_sigmoid_probs_and_classify
from myPack.eval.category_uncertainty import calculate_uncertainty_metrics


class SubjectPerformance:
    def __init__(self, ensemble_predictions: list, y_true: np.ndarray, logits):
        self._ensemble_predictions = np.array(ensemble_predictions)
        self._true_labels = y_true
        self._logits = logits

        # Predictions
        self._sigmoid_predictions = None
        self._final_predictions = None
        self._mean_predictions = None

        # Uncertainty
        self._predictive_entropy = None

        # Results
        self._majority = None
        self._samples_correct = None
        self._avg_samples = None
        self._model_prediction_accuracies = None

        # Calculate the above metrics
        self._calculate_metrics()

    # --------------
    # Properties
    # --------------

    @property
    def majority(self):
        return self._majority

    @property
    def mean_predictions(self):
        return self._mean_predictions

    @property
    def predictive_entropy(self):
        return self._predictive_entropy

    @property
    def sigmoid_predictions(self):
        return self._sigmoid_predictions

    # --------------
    # End Properties
    # --------------

    def _calculate_metrics(self) -> None:
        """ Calculates most of the general metrics from a subject:
                - Majority Voting for the ensemble
                - Average correctly predicted sample
                - Number of correctly predicted sample per model
                - Final predictions and output from sigmoid from all models
                - Predictive Entropy
                - Mean of predictions
        Returns:
            None, but saves them as variable in the object

        """
        sigmoid_list = list()
        final_prediction_list = list()
        prediction_acc = list()
        prediction_num = list()

        for pred in self._ensemble_predictions:
            output_from_sigmoid, binary_predictions = apply_sigmoid_probs_and_classify(pred, is_logits=self._logits)

            sigmoid_list.append(output_from_sigmoid)
            final_prediction_list.append(binary_predictions)

            acc = accuracy_score(y_true=self._true_labels, y_pred=binary_predictions)
            num_correct = accuracy_score(y_true=self._true_labels, y_pred=binary_predictions, normalize=False)

            prediction_acc.append(acc)
            prediction_num.append(num_correct)

        if np.mean(prediction_acc) > 0.5:
            self._majority = 1
        else:
            self._majority = 0

        self._model_prediction_accuracies = prediction_acc
        self._samples_correct = prediction_num
        self._avg_samples = np.mean(prediction_num)

        self._sigmoid_predictions = np.array(sigmoid_list)
        self._final_predictions = np.array(final_prediction_list)
        self._predictive_entropy, self._mean_predictions = \
            calculate_uncertainty_metrics(predictions=final_prediction_list, output_from_sigmoid=sigmoid_list)
