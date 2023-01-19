import numpy as np
import tensorflow as tf


def evaluate_majority_voting(model_object, test_dict: dict):
    subject_predicted_correctly = 0

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        eval_metrics = model_object.predict(data=data_x, labels=data_y, return_metrics=True)

        if eval_metrics['accuracy'] > 0.5:
            subject_predicted_correctly += 1

    return subject_predicted_correctly / len(test_dict.keys())


def evaluate_ensembles(model, test_dict: dict, write_file_path, monte_carlo_ensemble):
    num_monte_carlo = 10
    model_ensemble = False

    if isinstance(model, list):
        model_ensemble = True

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        pred_list = list()
        if model_ensemble:
            for m in model:
                _, _, temp_class = m.predict(data=data_x, monte_carlo=model_ensemble)
                pred_list.append(temp_class)
        else:
            data_x_tensor = tf.convert_to_tensor(data_x, dtype=tf.float64)
            for _ in range(num_monte_carlo):
                _, _, temp_class = model.predict(data=data_x_tensor, monte_carlo=True)
                pred_list.append(temp_class)

    # todo Two options - or do both
    # Evaluate each subject according to each of the members of the ensemble and how the majority votes
    # Or average each sample over the ensemble, and then do the majority??
