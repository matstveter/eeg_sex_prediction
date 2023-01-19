import numpy as np


def evaluate_majority_voting(model_object, test_dict: dict):

    subject_predicted_correctly = 0

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])

        eval_metrics = model_object.predict(data=data_x, labels=data_y, return_metrics=True)

        if eval_metrics['accuracy'] > 0.5:
            subject_predicted_correctly += 1

    return subject_predicted_correctly / len(test_dict.keys())


