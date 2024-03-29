import os
import keras.backend
import gc

from myPack.classifiers.model_chooser import get_model
from myPack.data.data_handler import create_k_folds, get_all_data_and_generators
from myPack.eval.performance_evaluation import evaluate_ensembles, evaluate_majority_voting
from myPack.utils import save_to_pkl, write_confidence_interval, write_to_file, load_keras_model


def complete_model_evaluation(model_object, test_set_generator, test_dict, write_file_path) -> dict:
    """ Function that evaluate the test set with both independent samples and majority voting, stores this to a dict,
    writes the results to the metrics txt file and returns the dictionary

    Args:
        model_object: model of type super_classifier.py
        test_set_generator: test set data generator
        test_dict: test set dictionary
        write_file_path: path to where the file where the results should be written

    Returns:
        dict with metrics
    """
    # Evaluate trained model vs. the independent test set, this function returns a dictionary
    eval_metrics = model_object.predict(data=test_set_generator, return_metrics=True)

    # Evaluate model with majority voting and store in eval_metrics which is a dictionary
    eval_metrics['majority_voting_accuracy'], male_correct, female_correct = evaluate_majority_voting(
        model_object=model_object, test_dict=test_dict)

    write_to_file(file_name_path=write_file_path, message=f"Test Set Accuracy: {eval_metrics['accuracy']}")
    write_to_file(file_name_path=write_file_path, message=f"Test Majority Accuracy: "
                                                          f"{eval_metrics['majority_voting_accuracy']}")
    write_to_file(file_name_path=write_file_path, message=f"Test Set AUC: {eval_metrics['auc']}")
    write_to_file(file_name_path=write_file_path, message=f"Test Set Male Correct: {male_correct}")
    write_to_file(file_name_path=write_file_path, message=f"Test Set Female Correct: {female_correct}")
    return eval_metrics


def run_final_experiments(data_dict: dict, model_dict: dict, hyper_dict: dict, time_dict: dict, general_dict: dict):
    """ This function performs the various experiments based on the experiment type. Starts by creating a list of the
    different subjects in the different lists. Write to file, initialize dictionaries and list for results.
    Depending on the experiments, loops through, creating new generators based on the k-fold, trains model, evaluates,
    save model, save results and re-iterate.

    If there are ensembles, the various models is saved, and after all models have been trained one k-fold, evaluate
    the ensemble and calculate uncertainty

    Tips for improvement:
        - Save models during training, so that each model does not have to be trained multiple times. For example:
        if InceptionTime is trained for "Single Model" experiment, use the same models for the model ensemble.

    Args:
        data_dict: Dictionary containing data and label information
        model_dict: Dictionary with model specific information, needed for model creation
        hyper_dict: Dictionary with hyperparameters
        time_dict: Dictionary with EEG specific information, such as epoch length
        general_dict: Usually paths and general stuff

    Returns:
        None
    """
    num_k_folds = 5
    num_models_in_ensemble = 5

    train_set_list, val_set_list, test_set_list = create_k_folds(data_dict=data_dict, num_folds=num_k_folds)

    write_to_file(general_dict['write_file_path'], message=f"**** Starting Experiment: "
                                                           f"{general_dict['experiment_type']} ****"
                                                           f"\nModel(s): {model_dict['model_name']}")
    metrics_dictionary = dict()
    ensemble_subject_accuracy = []
    ensemble_sample_accuracy = []

    if general_dict['experiment_type'] == "single_model":
        # Used for conf intervals
        accuracy = []
        majority_accuracy = []
        area_under_curve = []

        for n in range(num_k_folds):
            write_to_file(general_dict['write_file_path'], f"Starting Experiment with fold: {n + 1} / {num_k_folds}")

            # Load new generators based on the train_id,val_id, test_id which is the k-fold
            train_generator, val_generator, test_generator, test_set_dict, model_shape = \
                get_all_data_and_generators(data_dict=data_dict,
                                            time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=model_dict['use_conv2d'],
                                            only_test=general_dict['testing'],
                                            load_test_dict=True)
            model_dict['input_shape'] = model_shape

            # Get a new model
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=str(n))

            # Train model
            training_history = model_object.fit(train_generator=train_generator,
                                                validation_generator=val_generator,
                                                plot_test_acc=True, save_raw=True)

            # Evaluate the model on the test set
            eval_metrics = complete_model_evaluation(model_object=model_object,
                                                     test_set_generator=test_generator,
                                                     test_dict=test_set_dict,
                                                     write_file_path=general_dict['write_file_path'])
            eval_metrics['training_history'] = training_history.history

            # Store in lists for easier calculation of confidence intervals afterwards
            accuracy.append(eval_metrics['accuracy'])
            majority_accuracy.append(eval_metrics['majority_voting_accuracy'])
            area_under_curve.append(eval_metrics['auc'])

            model_object.model.save(general_dict['model_path'] + "/keras_model" + "_" + str(n))

            # If monte carlo dropout should be used
            if model_dict['apply_mc']:
                per_subject, per_sample, eff_windows, maj_windows = evaluate_ensembles(model=model_object,
                                                                                       test_dict=test_set_dict,
                                                                                       write_file_path=general_dict[
                                                                                           'write_file_path'],
                                                                                       figure_path=general_dict[
                                                                                                       'fig_path'] +
                                                                                                   model_dict[
                                                                                                       'model_name'] + "_" + str(
                                                                                           n))
                # Save ensemble metrics
                eval_metrics['ensemble_performance_subject'] = per_subject
                eval_metrics['ensemble_performance_sample'] = per_sample
                eval_metrics['effective_windows'] = eff_windows
                eval_metrics['effective_maj_windows'] = maj_windows
                ensemble_subject_accuracy.append(per_subject)
                ensemble_sample_accuracy.append(per_sample)

            metrics_dictionary["model_" + str(n)] = eval_metrics
            write_to_file(general_dict['write_file_path'], f"Ending Experiment with fold: {n + 1} / {num_k_folds}\n\n")
            _ = gc.collect()
            keras.backend.clear_session()
            test_set_dict = None

        # Saving data, and calculate confidence intervals from all the runs
        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_confidence_interval(metric=accuracy, write_file_path=general_dict['write_file_path'],
                                  metric_name="Accuracy")
        write_confidence_interval(metric=majority_accuracy, write_file_path=general_dict['write_file_path'],
                                  metric_name="Majority Voting")
        write_confidence_interval(metric=area_under_curve, write_file_path=general_dict['write_file_path'],
                                  metric_name="AUC")

        if model_dict['apply_mc']:
            # Calculate and write confidence intervals if monte carlo dropout was used
            write_to_file(general_dict['write_file_path'], f"\nConfidence Intervals Ensembles: ", also_print=False)
            write_confidence_interval(metric=ensemble_subject_accuracy,
                                      write_file_path=general_dict["write_file_path"], metric_name="Per Subject:")
            write_confidence_interval(metric=ensemble_sample_accuracy,
                                      write_file_path=general_dict["write_file_path"], metric_name="Per Sample:")
    elif general_dict["experiment_type"] == "ensemble_weights":
        for n in range(num_k_folds):
            write_to_file(general_dict['write_file_path'], f"Starting Experiment with fold: {n + 1} / {num_k_folds}")
            train_generator, val_generator, test_generator, test_set_dict, model_shape = \
                get_all_data_and_generators(data_dict=data_dict,
                                            time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=model_dict['use_conv2d'],
                                            only_test=general_dict['testing'],
                                            load_test_dict=True)
            model_dict['input_shape'] = model_shape

            # List of trained models in the ensemble
            model_list = []
            ensemble_metrics = dict()

            # Comparing models in ensemble
            accuracy = []
            majority_accuracy = []
            area_under_curve = []

            for m in range(num_models_in_ensemble):
                write_to_file(general_dict['write_file_path'],
                              f"Starting Model: {m + 1} / {num_models_in_ensemble}")
                # Get Model
                model_object = get_model(which_model=model_dict['model_name'],
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=str(n) + "_" + str(m))

                # Train model
                training_history = model_object.fit(train_generator=train_generator,
                                                    validation_generator=val_generator,
                                                    plot_test_acc=True, save_raw=True)

                # Evaluate the model on the test set
                eval_metrics = complete_model_evaluation(model_object=model_object,
                                                         test_set_generator=test_generator,
                                                         test_dict=test_set_dict,
                                                         write_file_path=general_dict['write_file_path'])
                eval_metrics['training_history'] = training_history.history

                # Store in lists for easier calculation of confidence intervals afterwards
                accuracy.append(eval_metrics['accuracy'])
                majority_accuracy.append(eval_metrics['majority_voting_accuracy'])
                area_under_curve.append(eval_metrics['auc'])

                model_list.append(model_object)
                ensemble_metrics[str(n) + "_" + str(m)] = eval_metrics

                write_to_file(general_dict['write_file_path'],
                              f"Ending Model: {m + 1} / {num_models_in_ensemble}\n\n")

            per_subject, per_sample, eff_windows, maj_windows = evaluate_ensembles(model=model_list,
                                                                                   test_dict=test_set_dict,
                                                                                   write_file_path=general_dict[
                                                                                       "write_file_path"],
                                                                                   figure_path=general_dict[
                                                                                                   "fig_path"] +
                                                                                               "ensemble_" + str(n))
            ensemble_metrics['ensemble_performance_subject'] = per_subject
            ensemble_metrics['ensemble_performance_sample'] = per_sample
            ensemble_metrics['effective_windows'] = eff_windows
            ensemble_metrics['effective_maj_windows'] = maj_windows

            ensemble_subject_accuracy.append(per_subject)
            ensemble_sample_accuracy.append(per_sample)

            # Save performance of single ensemble and calculate the confidence intervals
            metrics_dictionary['ensemble_' + str(n)] = ensemble_metrics
            write_confidence_interval(metric=accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Accuracy")
            write_confidence_interval(metric=majority_accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Majority Voting")
            write_confidence_interval(metric=area_under_curve, write_file_path=general_dict['write_file_path'],
                                      metric_name="AUC")
            write_to_file(general_dict['write_file_path'], f"Ending Experiment with fold: {n + 1} / {num_k_folds}\n\n")
            save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name=f"metrics_dictionary_run_{n}")

            _ = gc.collect()
            keras.backend.clear_session()
            test_set_dict = None

        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_to_file(general_dict['write_file_path'], f"\n\n\nConfidence Intervals Ensembles: ", also_print=False)
        write_confidence_interval(metric=ensemble_subject_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Subject:")
        write_confidence_interval(metric=ensemble_sample_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Sample:")
    elif general_dict["experiment_type"] == "depth_ensemble":

        for n in range(num_k_folds):
            write_to_file(general_dict['write_file_path'], f"Starting Experiment with fold: {n + 1} / {num_k_folds}")

            train_generator, val_generator, test_generator, test_set_dict, model_shape = \
                get_all_data_and_generators(data_dict=data_dict,
                                            time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=model_dict['use_conv2d'],
                                            only_test=general_dict['testing'],
                                            load_test_dict=True)
            model_dict['input_shape'] = model_shape

            # List of trained models in the ensemble
            model_list = []
            ensemble_metrics = dict()

            # Comparing models in ensemble
            accuracy = []
            majority_accuracy = []
            area_under_curve = []
            if model_dict['apply_mc']:
                depths = [2, 4, 6, 8, 10]
            else:
                depths = [3, 5, 7, 9, 11]
            for d in depths:
                write_to_file(general_dict['write_file_path'], f"Starting depth_ensemble: {d}")

                model_object = get_model(which_model=model_dict['model_name'],
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=str(n) + "_" + str(d),
                                         depth=d)

                # Train model
                training_history = model_object.fit(train_generator=train_generator,
                                                    validation_generator=val_generator,
                                                    plot_test_acc=True, save_raw=True)

                # Evaluate the model on the test set
                eval_metrics = complete_model_evaluation(model_object=model_object,
                                                         test_set_generator=test_generator,
                                                         test_dict=test_set_dict,
                                                         write_file_path=general_dict['write_file_path'])
                eval_metrics['training_history'] = training_history.history

                # Store in lists for easier calculation of confidence intervals afterwards
                accuracy.append(eval_metrics['accuracy'])
                majority_accuracy.append(eval_metrics['majority_voting_accuracy'])
                area_under_curve.append(eval_metrics['auc'])

                model_list.append(model_object)
                ensemble_metrics[str(n) + "_" + str(d)] = eval_metrics
                write_to_file(general_dict['write_file_path'], f"Ending depth_ensemble: {d}")

            per_subject, per_sample, eff_windows, maj_windows = evaluate_ensembles(model=model_list,
                                                                                   test_dict=test_set_dict,
                                                                                   write_file_path=general_dict[
                                                                                       "write_file_path"],
                                                                                   figure_path=general_dict[
                                                                                                   "fig_path"] +
                                                                                               "ensemble_" + str(n))
            ensemble_metrics['ensemble_performance_subject'] = per_subject
            ensemble_metrics['ensemble_performance_sample'] = per_sample
            ensemble_metrics['effective_windows'] = eff_windows
            ensemble_metrics['effective_maj_windows'] = maj_windows

            ensemble_subject_accuracy.append(per_subject)
            ensemble_sample_accuracy.append(per_sample)

            # Save performance of single ensemble and calculate the confidence intervals
            metrics_dictionary['ensemble_' + str(n)] = ensemble_metrics
            write_confidence_interval(metric=accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Accuracy")
            write_confidence_interval(metric=majority_accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Majority Voting")
            write_confidence_interval(metric=area_under_curve, write_file_path=general_dict['write_file_path'],
                                      metric_name="AUC")

            save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name=f"metrics_dictionary_run_{n}")
            write_to_file(general_dict['write_file_path'], f"Ending Experiment with fold: {n + 1} / {num_k_folds}\n\n")
            _ = gc.collect()
            keras.backend.clear_session()
            test_set_dict = None
        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_to_file(general_dict['write_file_path'], f"\n\n\nConfidence Intervals Ensembles: ", also_print=False)
        write_confidence_interval(metric=ensemble_subject_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Subject:")
        write_confidence_interval(metric=ensemble_sample_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Sample:")
    elif general_dict["experiment_type"] == "ensemble_models":

        models = model_dict['model_name'].split(",")

        for n in range(num_k_folds):
            write_to_file(general_dict['write_file_path'], f"Starting Experiment with fold: {n + 1} / {num_k_folds}")
            train_gen, val_gen, test_gen, test_set_dict, model_shape1 = \
                get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=False,
                                            only_test=general_dict['testing'])
            train_gen2D, val_gen2D, test_gen2D, _, model_shape2D = \
                get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=True,
                                            only_test=general_dict['testing'],
                                            load_test_dict=False)

            # List of trained models in the ensemble
            model_list = []
            ensemble_metrics = dict()

            # Comparing models in ensemble
            accuracy = []
            majority_accuracy = []
            area_under_curve = []

            for m in models:
                write_to_file(general_dict['write_file_path'], f"---- Starting Model {m}/ {models} ----")
                if "Net" in m:
                    print("Changing Generator to 2D")
                    train_generator = train_gen2D
                    val_generator = val_gen2D
                    test_generator = test_gen2D
                    model_dict['input_shape'] = model_shape2D
                else:
                    print("Changing Generator to 1D")
                    train_generator = train_gen
                    val_generator = val_gen
                    test_generator = test_gen
                    model_dict['input_shape'] = model_shape1

                model_object = get_model(which_model=m,
                                         model_dict=model_dict,
                                         hyper_dict=hyper_dict,
                                         general_dict=general_dict,
                                         model_name=str(n) + "_" + m)

                # Train model
                training_history = model_object.fit(train_generator=train_generator,
                                                    validation_generator=val_generator,
                                                    plot_test_acc=True, save_raw=True)

                # Evaluate the model on the test set
                eval_metrics = complete_model_evaluation(model_object=model_object,
                                                         test_set_generator=test_generator,
                                                         test_dict=test_set_dict,
                                                         write_file_path=general_dict['write_file_path'])
                eval_metrics['training_history'] = training_history.history

                # Store in lists for easier calculation of confidence intervals afterwards
                accuracy.append(eval_metrics['accuracy'])
                majority_accuracy.append(eval_metrics['majority_voting_accuracy'])
                area_under_curve.append(eval_metrics['auc'])

                model_list.append(model_object)
                ensemble_metrics[str(n) + "_" + m] = eval_metrics

                write_to_file(general_dict['write_file_path'], f"---- Ending Model {m}/ {models} ----")

            # Start Ensemble Performance Evaluation #
            per_subject, per_sample, eff_windows, maj_windows = evaluate_ensembles(model=model_list,
                                                                                   test_dict=test_set_dict,
                                                                                   write_file_path=general_dict[
                                                                                       "write_file_path"],
                                                                                   figure_path=general_dict[
                                                                                                   "fig_path"] +
                                                                                               "ensemble_" + str(n))
            ensemble_metrics['ensemble_performance_subject'] = per_subject
            ensemble_metrics['ensemble_performance_sample'] = per_sample
            ensemble_metrics['effective_windows'] = eff_windows
            ensemble_metrics['effective_maj_windows'] = maj_windows

            ensemble_subject_accuracy.append(per_subject)
            ensemble_sample_accuracy.append(per_sample)

            # Save performance of single ensemble and calculate the confidence intervals
            metrics_dictionary['ensemble_' + str(n)] = ensemble_metrics
            write_confidence_interval(metric=accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Accuracy")
            write_confidence_interval(metric=majority_accuracy, write_file_path=general_dict['write_file_path'],
                                      metric_name="Majority Voting")
            write_confidence_interval(metric=area_under_curve, write_file_path=general_dict['write_file_path'],
                                      metric_name="AUC")
            save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name=f"metrics_dictionary_run_{n}")
            write_to_file(general_dict['write_file_path'], f"Ending Experiment with fold: {n + 1} / {num_k_folds}\n\n")
            _ = gc.collect()
            keras.backend.clear_session()
            test_set_dict = None

        # Saving and calculating ensemble performance in terms of confidence intervals
        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_to_file(general_dict['write_file_path'], f"\n\n\nConfidence Intervals Ensembles: ", also_print=False)
        write_confidence_interval(metric=ensemble_subject_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Subject:")
        write_confidence_interval(metric=ensemble_sample_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Sample:")
    elif general_dict["experiment_type"] == "frequency_ensemble":
        # Load all 5 models, should be saved

        main_path = "/home/tvetern/datasets/freq/"
        # paths_to_models = ['theta', 'delta', 'alpha', 'lbeta', 'hbeta', 'gamma']
        paths_to_models = ['theta', 'delta', 'lbeta']


        # MAKE SURE THAT ALL MODELS ARE TRAINED ON THE SAME DATA AND TESTED ON THE SAME

        metrics_dictionary = dict()
        ensemble_subject_accuracy = []
        ensemble_sample_accuracy = []

        for n in range(num_k_folds):
            write_to_file(general_dict['write_file_path'], f"Starting Experiment with fold: {n + 1} / {num_k_folds}")
            model_list = []

            import numpy
            for p in paths_to_models:
                path = os.path.join(main_path, p)
                full_path = os.path.join(path, f"models/keras_model_{n}")
                model = load_keras_model(full_path)
                model_list.append(model)


            ensemble_metrics = dict()



            _, _, test_generator, test_set_dict, model_shape = \
                get_all_data_and_generators(data_dict=data_dict,
                                            time_dict=time_dict,
                                            batch_size=hyper_dict['batch_size'],
                                            train_id=train_set_list[n],
                                            val_id=val_set_list[n],
                                            test_id=test_set_list[n],
                                            use_conv2d=model_dict['use_conv2d'],
                                            only_test=general_dict['testing'],
                                            load_test_dict=True)
            model_dict['input_shape'] = model_shape

            # Comparing models in ensemble
            per_subject, per_sample, eff_windows, maj_windows = evaluate_ensembles(model=model_list,
                                                                                   test_dict=test_set_dict,
                                                                                   write_file_path=general_dict[
                                                                                       "write_file_path"],
                                                                                   figure_path=general_dict[
                                                                                                   "fig_path"] +
                                                                                               "ensemble_" + str(n),
                                                                                   freq=True)
            ensemble_metrics['ensemble_performance_subject'] = per_subject
            ensemble_metrics['ensemble_performance_sample'] = per_sample
            ensemble_metrics['effective_windows'] = eff_windows
            ensemble_metrics['effective_maj_windows'] = maj_windows

            ensemble_subject_accuracy.append(per_subject)
            ensemble_sample_accuracy.append(per_sample)

            # Save performance of single ensemble and calculate the confidence intervals
            metrics_dictionary['ensemble_' + str(n)] = ensemble_metrics
            save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name=f"metrics_dictionary_run_{n}")

            _ = gc.collect()
            keras.backend.clear_session()
            test_set_dict = None
            write_to_file(general_dict['write_file_path'], f"Ending Experiment with fold: {n + 1} / {num_k_folds}\n\n")

        save_to_pkl(data_dict=metrics_dictionary, path=general_dict['fig_path'], name="metrics_dictionary")
        write_to_file(general_dict['write_file_path'], f"\n\n\nConfidence Intervals Ensembles: ", also_print=False)
        write_confidence_interval(metric=ensemble_subject_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Subject:")
        write_confidence_interval(metric=ensemble_sample_accuracy,
                                  write_file_path=general_dict["write_file_path"], metric_name="Per Sample:")
    else:
        raise ValueError(f"Experiment type is not recognized : {general_dict['experiment_type']}!")