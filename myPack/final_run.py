import numpy as np

from myPack.eval.performance_evaluation import evaluate_ensembles, evaluate_majority_voting
from myPack.classifiers.model_chooser import get_model
from myPack.data.data_handler import get_all_data_and_generators
from myPack.utils import get_conf_interval, load_pkl, plot_confidence_interval, write_to_file, save_to_pkl


def run_final_experiment(data_dict: dict, model_dict: dict, hyper_dict: dict, time_dict: dict, general_dict: dict):
    # Get data generators, test_dictionary and model shape
    train_generator, validation_generator, test_generator, test_set_dictionary, model_shape = \
        get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
                                    use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
    model_dict['input_shape'] = model_shape

    # Write to file, important information
    write_to_file(general_dict['write_file_path'], message=f"**** Starting Experiment: "
                                                           f"{general_dict['experiment_type']} ****"
                                                           f"\nModel(s): {model_dict['model_name']}", also_print=False)

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
                                     model_name=model_dict['model_name'] + "_" + str(i))
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
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            if model_dict['apply_mc']:
                evaluate_ensembles(model=model_object, test_dict=test_set_dictionary,
                                   write_file_path=general_dict['write_file_path'],
                                   figure_path=general_dict['fig_path'] + model_dict['model_name'] + "_" + str(i))
            metrics_dictionary[model_object.save_name] = eval_metrics

            accs.append(eval_metrics['accuracy'])
            maj_accs.append(eval_metrics['majority_voting_acc'])
            aucs.append(eval_metrics['auc'])
            write_to_file(general_dict['write_file_path'], f"---- Ending Run {i + 1}/{general_dict['num_models']} ----",
                          also_print=True)

        # RUN FINISHED -> SAVING STUFF #
        write_to_file(general_dict['write_file_path'],"Confidence intervals for all runs",
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
        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
    elif general_dict['experiment_type'] == "ensemble_models":
        models = model_dict['model_name'].split(",")
        model_list = list()
        histories = list()
        for m in models:
            write_to_file(general_dict['write_file_path'], f"---- Starting Model {m}/ {models} ----",
                          also_print=True)
            model_object = get_model(which_model=m,
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=m)
            train_history = model_object.fit(train_generator=train_generator,
                                             validation_generator=validation_generator,
                                             plot_test_acc=True,
                                             save_raw=True)

            histories.append(train_history)
            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            model_list.append(model_object)
            write_to_file(general_dict['write_file_path'], f"---- Ending Model {m}/ {models} ----",
                          also_print=True)

    elif general_dict['experiment_type'] == "ensemble_weights":
        kernel_init = "random_uniform"
        # kernel_init = "random_normal"

        model_list = list()
        histories = list()

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
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            model_list.append(model_object)

        evaluate_ensembles(model=model_list, test_dict=test_set_dictionary,
                           write_file_path=general_dict['write_file_path'],
                           figure_path=general_dict['fig_path'])

        # Evaluate the list of uncertainties from evaluate_ensemble method??

    elif general_dict['experiment_type'] == "frequency_ensemble":

        metrics_dictionary = dict()
        histories = list()

        dataset_path = "/home/tvetern/datasets/numpy/"
        dataset_ext = ['delta/', 'theta/', 'alpha/', 'beta/', 'gamma/']
        different_models = False

        ensemble_models = [None for _ in range(len(dataset_ext))]

        for i, d in enumerate(dataset_ext):
            write_to_file(general_dict['write_file_path'],
                          f"---- Starting Dataset {d}/{dataset_ext} ----",
                          also_print=True)
            data_dict = load_pkl(dataset_path + d)
            train_generator, validation_generator, test_generator, test_set_dictionary, model_shape = \
                get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
            if different_models:
                if d == "delta/":
                    pass
                elif d == "theta/":
                    pass
                elif d == "alpha/":
                    pass
                elif d == "beta/":
                    pass
                elif d == "gamma/":
                    pass
                else:
                    raise ValueError("Unrecognized model")
            else:
                model_name = model_dict['model_name']

            current_best_acc = 0
            for n in range(general_dict['num_models']):
                write_to_file(general_dict['write_file_path'],
                              f"---- Starting Run {n + 1}/{general_dict['num_models']} ----", also_print=True)
                model_object = get_model(which_model=model_name,
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=model_name + "_" + str(n))
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
                                                               f"{eval_metrics['majority_voting_acc']}",
                              also_print=True)

                if eval_metrics['accuracy'] > current_best_acc:
                    ensemble_models[i] = model_object
                    current_best_acc = eval_metrics['accuracy']
                    write_to_file(general_dict['write_file_path'], f"New Best Models for {d}, run: {n}",also_print=False)

                write_to_file(general_dict['write_file_path'],
                              f"---- Ending Run {n + 1}/{general_dict['num_models']} ----",
                              also_print=True)
            write_to_file(general_dict['write_file_path'],
                          f"---- Ending Dataset {d}/{dataset_ext} ----",
                          also_print=True)

        data_dict = load_pkl("/home/tvetern/datasets/numpy/raw_no_change/")
        _, _, test_generator, test_set_dictionary, model_shape = \
            get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
                                        use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])

        evaluate_ensembles(model=ensemble_models, test_dict=test_set_dictionary,
                           write_file_path=general_dict['write_file_path'],
                           figure_path=general_dict['fig_path'])
    else:
        raise ValueError(f"Experiment type is not recognized : {general_dict['experiment_type']}!")
