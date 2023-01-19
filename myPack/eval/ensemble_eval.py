import numpy as np
import tensorflow as tf
from myPack.utils import write_to_file
from myPack.eval.category_uncertainty import CategorySampleUncertainty
from myPack.eval.subject_performance import SubjectPerformance


def eval_certain_subjects(res_list: list, num_subjects: int, write_file: str) -> None:
    """ Evaluates the metrics saved from the ensemble run, and calculates how the certain vs rejected accuracy and other
    metrics are

    Args:
        res_list(list): [num_subjects, [4]] where [4] = [acc_kept, num_kep, acc_rejected, num_rejected]
        num_subjects: number of alle subjects in the test set
        write_file: path to write file

    Returns:
        None
    """
    certain_subjects = 0
    certain_and_correct = 0
    rejected_subjects = 0
    rejected_and_correct = 0
    num_kept_list = list()
    num_rej_list = list()

    for res in res_list:
        num_kept, num_kept_correct, num_reject, num_reject_correct = res[0], res[1], res[2], res[3]

        if num_kept > int(num_subjects/4):
            certain_subjects += 1
            if num_kept_correct/num_kept > 0.5:
                certain_and_correct += 1
        else:
            rejected_subjects += 1
            if num_reject_correct/num_reject > 0.5:
                rejected_and_correct += 1

        num_kept_list.append(num_kept)
        num_rej_list.append(num_reject)

    if certain_subjects != 0:
        write_to_file(write_file, f"Certain Subjects: {certain_subjects}, of these {certain_and_correct} "
                                  f"(Acc: {(certain_and_correct/certain_subjects)*100}) were correct!",
                                  also_print=False)
    else:
        write_to_file(write_file, f"Certain Subjects: {certain_subjects}", also_print=False)

    if rejected_subjects != 0:
        write_to_file(write_file, f"Rejected Subjects: {rejected_subjects}, of these {rejected_and_correct} "
                                  f"(Acc: {(rejected_and_correct/rejected_subjects)*100}) were correct!",
                      also_print=True)
    else:
        write_to_file(write_file, f"Rejected Subjects: {rejected_subjects}", also_print=False)


def evaluate_ensemble_majority(model, test_dict: dict, write_file, which_model="loaded_weights") -> dict:
    """ Majority voting on the test-set from the test dictionary, the model list contains trained models for
    ensembles. Looping through the test dict, extract data, predicts the data across all models, calculate uncertainty
    between each prediction from the models, and then take the mean of the predictions. Then evaluate teh uncertainty
    given teh predictions, and save the data.

    Args:
        which_model:
        model: list of trained models, or a single model if it is meant for MC
        test_dict: subject id as keu, value contains data, labels, and sex, age
        write_file: path to the writing file

    Returns:
        dictionary containing all data
    """
    if type(model) == list:
        logits = model[0].logits
        write_to_file(write_file, "---------- Deep Ensembles ----------", also_print=False)
    else:
        write_to_file(write_file, "---------- Monte Carlo Ensemble ----------", also_print=False)
        monte_carlo_model = model.create_mc_model(which_model=which_model)
        logits = model.logits
    write_to_file(write_file, f"Model: {which_model}", also_print=False)

    result_dict = dict()
    num_subjects = len(list(test_dict.keys()))

    majority_voted_from_mc = 0
    attention_vector_predictions = 0
    single_category = list()
    sex_wrong = list()
    num_samples = 0

    num_per_threshold = list()
    num_cor_threshold_kept = list()
    num_cor_threshold_rej = list()
    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])
        label = value['label']
        num_samples = data_x.shape[0]

        temp_logits_list = list()

        # Is the model a list of ensemble models or is the one model which we should do monte carlo dropout
        if type(model) == list:
            for m in model:
                temp_logits = m.perform_prediction(data_x=data_x, which_model=which_model)
                temp_logits_list.append(temp_logits)
        else:
            test_x_tensor = tf.convert_to_tensor(data_x, dtype=tf.float64)
            for i in range(model.num_mc):
                temp_logits = monte_carlo_model.predict(x=test_x_tensor, batch_size=model.batch_size)
                temp_logits_list.append(temp_logits)

        sub_perf = SubjectPerformance(ensemble_predictions=temp_logits_list, y_true=data_y, logits=logits)
        majority_voted_from_mc += sub_perf.majority

        if sub_perf.majority != 1:
            sex_wrong.append(label)

        sub_unc = CategorySampleUncertainty(mean_prediction=sub_perf.mean_predictions,
                                            predictive_entropy=sub_perf.predictive_entropy,
                                            true_label=label)

        single_category.append(sub_unc.get_single_category(threshold=[0.8, 0.2], scale=False))
        sub_unc.attention_vector(scale=False)
        attention_vector_predictions += sub_unc.attention_prediction_correct
        num_certain_per_thresh, num_cor_per_thresh_kept, num_cor_per_thresh_rej = sub_unc.calculate_category_results()

        num_per_threshold.append(np.array(num_certain_per_thresh))
        num_cor_threshold_kept.append(np.array(num_cor_per_thresh_kept))
        num_cor_threshold_rej.append(np.array(num_cor_per_thresh_rej))

    num_avg = np.mean(np.array(num_per_threshold), axis=0)
    num_avg_percentage = num_avg/num_samples
    cor_avg = np.mean(np.array(num_cor_threshold_kept), axis=0)
    certain_correct_avg = cor_avg/num_avg
    rej_avg = np.mean(np.array(num_cor_threshold_rej), axis=0)

    write_to_file(write_file, f"Majority Voting from Ensemble: {(majority_voted_from_mc / num_subjects) * 100}",
                  also_print=False)
    eval_certain_subjects(res_list=single_category, num_subjects=num_subjects, write_file=write_file)
    write_to_file(write_file, f"Attention Vector from Ensemble: {(attention_vector_predictions / num_subjects) * 100}",
                  also_print=False)
    write_to_file(write_file, f"Num Wrong: {len(sex_wrong)} -> Female wrong: {(sex_wrong.count(1)/len(sex_wrong))*100}, "
                              f"Male: {(sex_wrong.count(0)/len(sex_wrong))*100}", also_print=False)
    write_to_file(write_file, f"Avg samples per threshold: {num_avg} -> (%{num_avg_percentage})"
                              f"\nNum Correct Per Threshold of the kept samples: {cor_avg} -> (%{certain_correct_avg})"
                              f"\nNum correct of the rejected samples(should be as low as possible): {rej_avg}",
                  also_print=False)
    return result_dict
