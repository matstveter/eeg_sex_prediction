import numpy as np

from myPack.eval.performance_evaluation import evaluate_ensembles, evaluate_majority_voting
from myPack.classifiers.model_chooser import get_model
from myPack.data.data_handler import get_all_data_and_generators, k_fold_generators
from myPack.utils import get_conf_interval, load_pkl, plot_confidence_interval, write_to_file, save_to_pkl


def run_final_experiment(data_dict: dict, model_dict: dict, hyper_dict: dict, time_dict: dict, general_dict: dict):
    # Get data generators, test_dictionary and model shape
    # train_generator, validation_generator, test_generator, test_set_dictionary, model_shape = \
    #     get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
    #                                 use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
    # model_dict['input_shape'] = model_shape

    # Write to file, important information
    write_to_file(general_dict['write_file_path'], message=f"**** Starting Experiment: "
                                                           f"{general_dict['experiment_type']} ****"
                                                           f"\nModel(s): {model_dict['model_name']}", also_print=False)
    num_ensembles = 5

    if general_dict['experiment_type'] == "single_model":
        # Depending on the model that is running, this can adjust the models "extra" parameters

        metrics_dictionary = dict()
        histories = list()
        accs = list()
        maj_accs = list()
        aucs = list()

        for i in range(general_dict['num_models']):
            write_to_file(general_dict['write_file_path'],
                          f"---- Starting Run {i + 1}/{general_dict['num_models']} ----",
                          also_print=True)
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + "_" + str(i), depth=4)
            train_hist = model_object.fit(train_generator=train_generator,
                                          validation_generator=validation_generator,
                                          plot_test_acc=True,
                                          save_raw=True)
            histories.append(train_hist)
            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}"
                                                           f"\nTest AUC: {eval_metrics['auc']}", also_print=True)
            if model_dict['apply_mc']:
                per_sub, per_samp = evaluate_ensembles(model=model_object, test_dict=test_set_dictionary,
                                                       write_file_path=general_dict['write_file_path'],
                                                       figure_path=general_dict['fig_path'] + model_dict[
                                                           'model_name'] + "_" + str(i))
                eval_metrics['per_subject_mc'] = per_sub
                eval_metrics['per_sample_mc'] = per_samp
            metrics_dictionary[model_object.save_name] = eval_metrics

            accs.append(eval_metrics['accuracy'])
            maj_accs.append(eval_metrics['majority_voting_acc'])
            aucs.append(eval_metrics['auc'])
            write_to_file(general_dict['write_file_path'], f"---- Ending Run {i + 1}/{general_dict['num_models']} ----",
                          also_print=True)

        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        # RUN FINISHED -> SAVING STUFF #
        write_to_file(general_dict['write_file_path'], "Confidence intervals for all runs",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Accuracy (mean, lower, upper): {get_conf_interval(accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Majority Accuracy (mean, lower, upper): "
                                                       f"{get_conf_interval(maj_accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"AUC (mean, lower, upper): {get_conf_interval(aucs)}",
                      also_print=False)

        plot_confidence_interval(histories=histories, key="accuracy", save_name=general_dict['fig_path'])
        plot_confidence_interval(histories=histories, key="f1", save_name=general_dict['fig_path'])
        plot_confidence_interval(histories=histories, key="auc", save_name=general_dict['fig_path'])
    elif general_dict['experiment_type'] == "ensemble_models":
        models = model_dict['model_name'].split(",")

        ensemble_performance_subject = list()
        ensemble_performance_sample = list()

        for e in range(num_ensembles):
            write_to_file(general_dict['write_file_path'], f"Starting ensemble {e + 1} / {num_ensembles}")
            model_list = list()
            for m in models:
                write_to_file(general_dict['write_file_path'], f"---- Starting Model {m}/ {models} ----",
                              also_print=True)
                model_object = get_model(which_model=m,
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=m)
                model_object.fit(train_generator=train_generator,
                                 validation_generator=validation_generator,
                                 plot_test_acc=True,
                                 save_raw=True)

                eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
                eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                               test_dict=test_set_dictionary)

                write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                               f"\nMajority Voting Acc: "
                                                               f"{eval_metrics['majority_voting_acc']}"
                              f"\nTest AUC: {eval_metrics['auc']}", also_print=True)
                model_list.append(model_object)
                write_to_file(general_dict['write_file_path'], f"---- Ending Model {m}/ {models} ----",
                              also_print=True)
            sub_perf, samp_per = evaluate_ensembles(model=model_list, test_dict=test_set_dictionary,
                                                    write_file_path=general_dict['write_file_path'],
                                                    figure_path=general_dict['fig_path'] + "ensemble_" + str(e))
            ensemble_performance_subject.append(sub_perf)
            ensemble_performance_sample.append(samp_per)

            write_to_file(general_dict['write_file_path'], f"Ending ensemble {e + 1} / {num_ensembles}")

        # Evaluate the list of uncertainties from evaluate_ensemble method??
        write_to_file(general_dict['write_file_path'], "Confidence intervals for all ensembles",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Per Subject (mean, lower, upper): "
                                                       f"{get_conf_interval(ensemble_performance_subject)}")
        write_to_file(general_dict['write_file_path'], f"Per Subject (mean, lower, upper): "
                                                       f"{get_conf_interval(ensemble_performance_subject)}")

    elif general_dict['experiment_type'] == "ensemble_weights":
        kernel_init = "random_uniform"
        # kernel_init = "random_normal"

        ensemble_performance_subject = list()
        ensemble_performance_sample = list()

        for e in range(num_ensembles):
            write_to_file(general_dict['write_file_path'], f"Starting ensemble {e + 1} / {num_ensembles}")
            model_list = list()
            for i in range(general_dict['num_models']):
                write_to_file(general_dict['write_file_path'],
                              f"---- Starting Run {i + 1}/{general_dict['num_models']} ----",
                              also_print=True)

                model_object = get_model(which_model=model_dict['model_name'],
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=model_dict['model_name'] + "_" + str(i),
                                         kernel_init=kernel_init)

                model_object.fit(train_generator=train_generator,
                                 validation_generator=validation_generator,
                                 plot_test_acc=True,
                                 save_raw=True)
                eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
                eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                               test_dict=test_set_dictionary)

                write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                               f"\nMajority Voting Acc: "
                                                               f"{eval_metrics['majority_voting_acc']}"
                              f"\nTest AUC: {eval_metrics['auc']}", also_print=True)
                model_list.append(model_object)
                write_to_file(general_dict['write_file_path'],
                              f"---- Ending Run {i + 1}/{general_dict['num_models']} ----",
                              also_print=True)

            sub_perf, samp_per = evaluate_ensembles(model=model_list, test_dict=test_set_dictionary,
                                                    write_file_path=general_dict['write_file_path'],
                                                    figure_path=general_dict['fig_path'] + "ensemble_" + str(e))
            ensemble_performance_subject.append(sub_perf)
            ensemble_performance_sample.append(samp_per)

            write_to_file(general_dict['write_file_path'], f"Ending ensemble {e + 1} / {num_ensembles}")

        # Evaluate the list of uncertainties from evaluate_ensemble method??
        write_to_file(general_dict['write_file_path'], "Confidence intervals for all ensembles",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Per Subject (mean, lower, upper): "
                                                       f"{get_conf_interval(ensemble_performance_subject)}")
        write_to_file(general_dict['write_file_path'], f"Per Subject (mean, lower, upper): "
                                                       f"{get_conf_interval(ensemble_performance_subject)}")

    elif general_dict['experiment_type'] == "frequency_ensemble":

        metrics_dictionary = dict()
        accs = list()
        maj_accs = list()
        aucs = list()

        best_results = 0
        for i in range(general_dict['num_models']):
            write_to_file(general_dict['write_file_path'],
                          f"---- Starting Run {i + 1}/{general_dict['num_models']} ----",
                          also_print=True)
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + "_" + str(i), depth=4)
            train_hist = model_object.fit(train_generator=train_generator,
                                          validation_generator=validation_generator,
                                          plot_test_acc=True,
                                          save_raw=True)
            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}"
                                                           f"\nTest AUC: {eval_metrics['auc']}", also_print=True)
            metrics_dictionary[model_object.save_name] = eval_metrics

            if eval_metrics['majority_voting_acc'] > best_results:
                best_results = eval_metrics['majority_voting_acc']

                model_object.model.save(general_dict['model_path'])

            accs.append(eval_metrics['accuracy'])
            maj_accs.append(eval_metrics['majority_voting_acc'])
            aucs.append(eval_metrics['auc'])
            write_to_file(general_dict['write_file_path'], f"---- Ending Run {i + 1}/{general_dict['num_models']} ----",
                          also_print=True)
        # RUN FINISHED -> SAVING STUFF #
        write_to_file(general_dict['write_file_path'], "Confidence intervals for all runs",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Accuracy (mean, lower, upper): {get_conf_interval(accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Majority Accuracy (mean, lower, upper): "
                                                       f"{get_conf_interval(maj_accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"AUC (mean, lower, upper): {get_conf_interval(aucs)}",
                      also_print=False)

    elif general_dict['experiment_type'] == "k_fold":
        num_k_folds = 5

        train_generator_list, validation_generator_list, test_generator, test_set_dictionary, model_shape = \
            k_fold_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
                              num_folds=num_k_folds, use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
        model_dict['input_shape'] = model_shape

        histories = list()
        accs = list()
        maj_accs = list()
        aucs = list()
        metrics_dictionary = dict()

        for i in range(num_k_folds):
            write_to_file(general_dict['write_file_path'],
                          f"---- Starting Run {i + 1}/ {num_k_folds} ----",
                          also_print=True)
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + "_" + str(i), depth=4)
            train_hist = model_object.fit(train_generator=train_generator_list[i],
                                          validation_generator=validation_generator_list[i],
                                          plot_test_acc=True,
                                          save_raw=True)
            histories.append(train_hist)

            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}"
                                                           f"\nTest AUC: {eval_metrics['auc']}", also_print=True)
            if model_dict['apply_mc']:
                per_sub, per_samp = evaluate_ensembles(model=model_object, test_dict=test_set_dictionary,
                                                       write_file_path=general_dict['write_file_path'],
                                                       figure_path=general_dict['fig_path'] + model_dict[
                                                           'model_name'] + "_" + str(i))
                eval_metrics['per_subject_mc'] = per_sub
                eval_metrics['per_sample_mc'] = per_samp
            metrics_dictionary[model_object.save_name] = eval_metrics

            accs.append(eval_metrics['accuracy'])
            maj_accs.append(eval_metrics['majority_voting_acc'])
            aucs.append(eval_metrics['auc'])
            write_to_file(general_dict['write_file_path'], f"---- Ending Run {i + 1}/{num_k_folds} ----",
                          also_print=True)

        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_to_file(general_dict['write_file_path'], "Confidence intervals for all runs",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Accuracy (mean, lower, upper): {get_conf_interval(accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"Majority Accuracy (mean, lower, upper): "
                                                       f"{get_conf_interval(maj_accs)}",
                      also_print=False)
        write_to_file(general_dict['write_file_path'], f"AUC (mean, lower, upper): {get_conf_interval(aucs)}",
                      also_print=False)

        plot_confidence_interval(histories=histories, key="accuracy", save_name=general_dict['fig_path'])
        plot_confidence_interval(histories=histories, key="f1", save_name=general_dict['fig_path'])
        plot_confidence_interval(histories=histories, key="auc", save_name=general_dict['fig_path'])
    else:
        raise ValueError(f"Experiment type is not recognized : {general_dict['experiment_type']}!")
