import enlighten
import numpy as np
import tensorflow as tf
import keras.backend
from sklearn.metrics import accuracy_score

from myPack.utils import write_to_file


def evaluate_majority_voting(model_object, test_dict: dict):
    subject_predicted_correctly = 0

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        eval_metrics = model_object.predict(data=data_x, labels=data_y, return_metrics=True)

        if eval_metrics['accuracy'] > 0.5:
            subject_predicted_correctly += 1

    return subject_predicted_correctly / len(list(test_dict.keys()))


def evaluate_ensembles(model, test_dict: dict, write_file_path: str):
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

    pbar = enlighten.Counter(total=len(list(test_dict.keys())), desc="Number of participants", unit="Participants")
    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        pred_list = list()
        class_probabilites = list()
        if model_ensemble:
            for m in model:
                _, probs, temp_class = m.predict(data=data_x)
                pred_list.append(temp_class)
                class_probabilites.append(probs)
        else:
            data_x_tensor = tf.convert_to_tensor(data_x, dtype=tf.float64)
            for _ in range(num_monte_carlo):
                _, probs, temp_class = model.predict(data=data_x_tensor, monte_carlo=True)
                pred_list.append(temp_class)
                class_probabilites.append(probs)

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

        # evaluate_uncertainty_per_subject(ensemble_predicted_probabilities=class_probabilites,
        #                                  mean_predictions=avg_ensemble_classes)

        break
    # Printing #
    write_to_file(write_file_path, f"Ensemble per subject: {ensemble_subject_majority / len(list(test_dict.keys()))}",
                  also_print=True)
    write_to_file(write_file_path, f"Ensemble per sample: {ensemble_sample_majority / len(list(test_dict.keys()))}",
                  also_print=True)

    return ensemble_subject_majority, ensemble_sample_majority


def calculate_uncertainty_metrics(sigmoid_predictions):
    # Transforming sigmoid from a single number, to a probability distribution over the classes
    prob_pos = np.copy(np.mean(sigmoid_predictions, axis=0))
    prob_neg = 1 - prob_pos
    class_prob_distribution = np.column_stack((prob_neg, prob_pos))

    variance = np.var(sigmoid_predictions, axis=0)
    pred_entropy = ((-np.sum(class_prob_distribution * np.log(class_prob_distribution + 1e-12), axis=-1)) / np.log(2))

    return None


def evaluate_uncertainty_per_subject(ensemble_predicted_probabilities: list, mean_predictions):
    calculate_uncertainty_metrics(sigmoid_predictions=ensemble_predicted_probabilities)
