import enlighten
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend
from sklearn.metrics import accuracy_score

from myPack.classifiers.keras_utils import apply_sigmoid_probs_and_classify
from myPack.utils import create_folder, write_to_file


def evaluate_majority_voting(model_object, test_dict: dict):
    subject_predicted_correctly = 0
    male_correct = 0
    female_correct = 0

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        eval_metrics = model_object.predict(data=data_x, labels=data_y, return_metrics=True)

        if eval_metrics['accuracy'] > 0.5:
            subject_predicted_correctly += 1

            if value['label'] == 0:
                male_correct += 1
            else:
                female_correct += 1

    return subject_predicted_correctly / len(list(test_dict.keys())), male_correct/ int(len(list(test_dict.keys()))/2), \
        female_correct / int(len(list(test_dict.keys()))/2)


def evaluate_ensembles(model, test_dict: dict, write_file_path: str, figure_path: str, freq=False):
    """
    * Receives a model, which can be of type SuperClassifer model, or a list of these.
    * Predicts according to the last point, if it is a model, MC Dropout predictions will be used, else just ensemble
    * Calculate two metrics, and print them to file
                1. sample(mean across each sample)
                2. subject(mean across the predictions for all windows for all ensembles)

    Args:
        model: list(modelObject) or modelObject
        test_dict: dictionary containing data and labels for each class
        write_file_path: path to the result txt file
        figure_path: where to save the figures

    Returns:
        ensemble_sample_majority
        ensemble_subject_majority
    """

    num_monte_carlo = 50
    model_ensemble = False

    if isinstance(model, list):
        model_ensemble = True

    ensemble_sample_majority = 0
    ensemble_subject_majority = 0

    pbar = enlighten.Counter(total=len(list(test_dict.keys())), desc="Performance and Uncertainty Evaluation",
                             unit="Participants")

    window_eval_dict = {'1.0': 0, '0.9': 0, '0.75': 0, '0.5': 0, '0.25': 0, '0.1': 0}
    window_maj_dict = {'1.0': 0, '0.9': 0, '0.75': 0, '0.5': 0, '0.25': 0, '0.1': 0}

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        pred_list = list()
        class_probabilities = list()
        auc_list = list()
        if model_ensemble:
            for m in model:
                # Ensures that EEGNet and InceptionTime receives the correct shape
                if freq:
                    y_pred = m.predict(x=data_x, batch_size=10)
                    probs, temp_class = apply_sigmoid_probs_and_classify(y_prediction=y_pred, is_logits=False)
                    pred_list.append(temp_class)
                    class_probabilities.append(probs)
                else:
                    if "Net" in m.save_name:
                        temp_x = np.expand_dims(data_x, axis=3)
                    else:
                        temp_x = data_x
                    _, probs, temp_class = m.predict(data=temp_x)
                    pred_list.append(temp_class)
                    class_probabilities.append(probs)
        else:
            data_x_tensor = tf.convert_to_tensor(data_x, dtype=tf.float64)
            for _ in range(num_monte_carlo):
                _, probs, temp_class = model.predict(data=data_x_tensor, monte_carlo=True)
                pred_list.append(temp_class)
                class_probabilities.append(probs)

        # Calculating the mean of the predictions (n_ensemble, windows, 1) -> (windows, 1)
        avg_ensemble_pred = np.mean(pred_list, axis=0)

        # The mean will return a np.array with floats -> round to make it binary
        avg_ensemble_classes = keras.backend.round(avg_ensemble_pred)

        # Loops through, checks accuracy of each of the models in the ensemble and saves the accuracy in the list
        accs = list()
        for p in pred_list:
            acc = accuracy_score(y_true=data_y, y_pred=p)
            accs.append(acc)

        # Each of the ensemble predict one sample, avg, then compare with label
        per_sample_ensemble_accuracy = accuracy_score(y_true=data_y, y_pred=avg_ensemble_classes)

        # Each subject predicts all windows from a subject, and then the accuracies of each ensemble is averaged
        per_subject_ensemble_accuracy = np.mean(accs, axis=0)

        # If the accuracy is more than 0.5 the subject is predicted as correctly
        if per_subject_ensemble_accuracy > 0.5:
            ensemble_subject_majority += 1

        # If the accuracy is more than 0.5 the subject is predicted as correctly
        if per_sample_ensemble_accuracy > 0.5:
            ensemble_sample_majority += 1

        t_dict, maj_dict = evaluate_effective_windows(sigmoid_ensemble_predictions=class_probabilities,
                                                      prediction=avg_ensemble_classes, label_sub=value['label'],
                                                      keys=list(window_eval_dict.keys()))

        for k in window_eval_dict.keys():
            window_eval_dict[k] += t_dict[k]

        for k in window_maj_dict.keys():
            window_maj_dict[k] += maj_dict[k]

        pbar.update()
    # Printing #
    per_subject_acc = ensemble_subject_majority / len(list(test_dict.keys()))
    per_sample_acc = ensemble_sample_majority / len(list(test_dict.keys()))

    write_to_file(write_file_path, f"Ensemble per-subject: {per_subject_acc}", also_print=True)
    write_to_file(write_file_path, f"Ensemble per-sample: {per_sample_acc}", also_print=True)

    plot_window_selection_performance(window_dict=window_eval_dict.copy(),
                                      total_number_subjects=len(list(test_dict.keys())),
                                      figure_path=figure_path+"acc")
    plot_window_selection_performance(window_dict=window_maj_dict.copy(),
                                      total_number_subjects=len(list(test_dict.keys())),
                                      figure_path=figure_path+"maj")
    for k, v in window_eval_dict.items():
        window_eval_dict[k] = (v / len(list(test_dict.keys()))) * 100

    for k, v in window_maj_dict.items():
        window_maj_dict[k] = (v / len(list(test_dict.keys()))) * 100

    write_to_file(write_file_path, f"Accuracy Window Uncertainty: {window_eval_dict}", also_print=True)
    write_to_file(write_file_path, f"Majority Window Uncertainty: {window_maj_dict}", also_print=True)

    return per_subject_acc, per_sample_acc, window_eval_dict, window_maj_dict


def plot_window_selection_performance(window_dict: dict, total_number_subjects, figure_path):
    for k, v in window_dict.items():
        window_dict[k] = (v / total_number_subjects) * 100

    plt.plot(list(window_dict.keys()), list(window_dict.values()), label="Removed Windows")
    plt.title("Evaluation of removing uncertain windows")
    plt.xlabel("Percentage Kept Windows")
    plt.ylabel("Accuracy on test set")
    plt.savefig(figure_path + "_eff_window.png")
    plt.close()


def evaluate_effective_windows(sigmoid_ensemble_predictions, prediction, label_sub: int,
                               keys) -> [dict, list]:
    """
    Receives a sigmoid ensemble prediction list [n_models, num_windows, 1], calculate the variance of each prediction
    over the samples. This will give a indication of which of the windows do the ensemble have different opinion about.
    Sort the ensemble prediction list according to this uncertainty variance list.

    Loop through and remove different amounts of windows, starting with removing 10% of the most uncertain windows.
    Continuously evaluate the remaining value against the label, and store 1 in a dict if the majority predictions of
    the remaining windows were correct, else 0.


    Args:
        sigmoid_ensemble_predictions: [num_ensemble_models, windows, 1] sigmoid output from the ensemble
        prediction: Binary list created from the sigmoid output
        label_sub: 1 if female, 0 if male.

    Returns:
        dictionary of the results from each of the amount of kept windows
    """
    # Calculate the simplest metrics for uncertainty -> variance across the ensemble on the samples
    keep_percentage_windows = [float(i) for i in keys]
    uncertain_samples = np.var(sigmoid_ensemble_predictions, axis=0)

    zipped_list = zip(list(uncertain_samples), list(prediction))
    sorted_list = sorted(zipped_list, key=lambda x: x[0])
    uncertain_samples, prediction = zip(*sorted_list)

    res_dict = dict()
    maj_dict = dict()

    # Loop through the list of the percentage of windows to be removed and eval the remaining windows and store in dict
    for k_percentage in keep_percentage_windows:
        temp_pred = prediction[0: int(len(prediction) * k_percentage)]
        lab = [label_sub] * len(temp_pred)

        acc = accuracy_score(y_true=lab, y_pred=temp_pred)
        # If accuracy is above 0.5, meaning that more than half the samples were predicted correctly, save 1
        if acc > 0.5:
            maj_dict[str(k_percentage)] = 1
        else:
            maj_dict[str(k_percentage)] = 0
        res_dict[str(k_percentage)] = acc

    return res_dict, maj_dict


# SUPPORTING FUNCTIONS FOR THE MAIN FUNCTION ABOVE #

def get_predictive_entropy(probs: np.ndarray):
    """ Calculated the predictive entropy from either a list of [1, 2] vectors or just a [1,2] vector
    Args:
        probs: numpy array with shape [1, 2] or more [N, 2]

    Returns:
        float: predictive entropy
    """
    return (-np.sum(probs * np.log(probs + 1e-12), axis=-1)) / np.log(2)


def calculate_uncertainty_metrics(sigmoid_predictions):
    """
    Calculates variance and predictive entropy from the sigmoid predictions, the entropy expects a probability
    distribution over the classes, so the sigmoid is transformed like this [class0, class1] = [1-sigmoid, sigmoid] and
    then calculates the predictive entropy

    Args:
        sigmoid_predictions: Output from sigmoid (num_models, num_windows, 1)

    Returns:
        variance (num_windows, 1)
        pred_entropy (num_windows, 1)
    """
    # Transforming sigmoid from a single number, to a probability distribution over the classes
    prob_pos = np.copy(np.mean(sigmoid_predictions, axis=0))
    prob_neg = 1 - prob_pos
    class_prob_distribution = np.column_stack((prob_neg, prob_pos))

    variance = np.var(sigmoid_predictions, axis=0)
    pred_entropy = get_predictive_entropy(class_prob_distribution)

    return variance, pred_entropy


class UncertaintyPerSubject:

    def __init__(self, sub_id, ensemble_predicted_probabilities: list, figure_path: str,
                 ensemble_sample_correct: float, ensemble_subject_correct: float):
        self._sub_id = sub_id

        self._ensemble_predicted_probabilities = ensemble_predicted_probabilities
        self._predicted_classes = keras.backend.round(np.mean(ensemble_predicted_probabilities, axis=0))
        self._variance, self._predictive_entropy = calculate_uncertainty_metrics(ensemble_predicted_probabilities)

        self._category_threshold = [np.array([1.0, 0.0]), np.array([0.95, 0.05]), np.array([0.9, 0.1]),
                                    np.array([0.8, 0.2]), np.array([0.7, 0.3]), np.array([0.6, 0.4]),
                                    np.array([0.5, 0.5])]
        self._entropy_thresholds = [get_predictive_entropy(probs=c_prob) for c_prob in self._category_threshold]
        self._figure_path = figure_path
        self._class_distribution = None
        self._get_class_distribution()

        # Calculated outside the class
        self._ensemble_sample_correct = ensemble_sample_correct
        self._ensemble_subject_correct = ensemble_subject_correct

        self._predictions_barplot()
        self._get_window_prediction_distribution()

    # ----------------#
    # Properties      #
    # ----------------#
    @property
    def variance(self):
        return self._variance

    @property
    def predictive_entropy(self):
        return self._predictive_entropy

    # ----------------#
    # End Properties  #
    # ----------------#

    def _get_class_distribution(self):
        prob_pos = np.copy(np.mean(self._ensemble_predicted_probabilities, axis=0))
        prob_neg = 1 - prob_pos
        self._class_distribution = np.column_stack((prob_neg, prob_pos))

    def _predictions_barplot(self):
        """
        Calculates the number of predicted classes from the average of the ensemble, and creates a barplot showing
        how man windows were predicted as 0 and 1.

        Returns:
            classes: list of number of windows predicted as class0 or class1
        """
        female = list(self._predicted_classes).count(1)
        male = list(self._predicted_classes).count(0)
        classes = [male, female]
        bars = ['Male', 'Female']
        plt.bar(bars, classes, color=['#ff7f0e', '#1f77b4'])
        plt.xlabel("Classes")
        plt.ylabel("Num windows")
        plt.title("Predictions")
        plt.tight_layout()
        plt.savefig(self._figure_path + self._sub_id + "_class_predictions.png")
        plt.close()
        return classes

    def _get_window_prediction_distribution(self):
        """
        Creates a bar-plot with the uncertainty distribution according to the predictive entropy and saves
        according to the figure path argument sent to the init path

        Returns:
            counter: number of windows per threshold
            ranges: ranges
        """
        entropy_ranges = []
        entropy_counts = [0 for _ in range(len(self._entropy_thresholds) - 1)]
        male_counts = [0 for _ in range(len(self._entropy_thresholds) - 1)]
        female_counts = [0 for _ in range(len(self._entropy_thresholds) - 1)]
        barpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bar_names = ['Very \nCertain', 'Certain', 'Moderat \nCertain', 'Uncertain', 'Very \nUncertain', 'Guessing']

        for i in range(len(self._predictive_entropy)):
            for j in range(len(self._entropy_thresholds) - 1):
                if self._entropy_thresholds[j] <= self._predictive_entropy[i] < self._entropy_thresholds[j + 1]:
                    entropy_ranges.append((self._entropy_thresholds[j], self._entropy_thresholds[j + 1]))
                    entropy_counts[j] += 1

                    if self._predicted_classes[i] == 1:
                        female_counts[j] += 1
                    else:
                        male_counts[j] += 1
                    break

        assert sum(male_counts) + sum(female_counts) == sum(entropy_counts), "Missing data, number of female + male " \
                                                                             "does not add up!"
        X_axis = np.arange(len(bar_names))

        plt.bar(X_axis - 0.2, female_counts, 0.4, label="Female")
        plt.bar(X_axis + 0.2, male_counts, 0.4, label="Male")
        plt.xticks(X_axis, bar_names)
        plt.title("Certainty Distribution")
        plt.xlabel("Degree of certain")
        plt.ylabel("Num windows")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(self._figure_path + self._sub_id + "_certain_dist.png")
        plt.close()

        plt.bar(bar_names, entropy_counts, color=barpl_colors)
        plt.xlabel("Degree of certain")
        plt.ylabel("Num windows")
        plt.title("Certainty Distribution")
        plt.tight_layout()
        plt.savefig(self._figure_path + self._sub_id + "_certain_all.png")
        plt.close()

        return entropy_ranges, entropy_counts
