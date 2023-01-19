import keras
import tensorflow as tf

from myPack.classifiers.model_chooser import get_model
from myPack.data.data_handler import get_all_data_and_generators
from myPack.eval.performance_evaluation import evaluate_majority_voting
from myPack.utils import write_to_file
from myPack.classifiers.keras_utils import mcc, f1, specificity, recall, precision


def testing_models(data_dict, model_dict, hyper_dict, time_dict, general_dict):
    train_generator, validation_generator, test_generator, test_set_dictionary, model_shape = \
        get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
                                    use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
    model_dict['input_shape'] = model_shape

    # Write to file, important information
    write_to_file(general_dict['write_file_path'], message=f"**** Starting Experiment: "
                                                           f"{general_dict['experiment_type']} ****"
                                                           f"\nModel(s): {model_dict['model_name']}", also_print=False)

    if general_dict['experiment_type'] == "depth":
        depths = [2, 4, 6, 8]
        for d in depths:
            write_to_file(general_dict['write_file_path'], f"---- Starting Depth {d}/ {depths} ----",
                          also_print=True)
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + f"_depth_{d}",
                                     depth=d)
            _ = model_object.fit(train_generator=train_generator,
                                 validation_generator=validation_generator,
                                 plot_test_acc=True,
                                 save_raw=False)

            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            write_to_file(general_dict['write_file_path'], f"---- Ending Depth {d}/ {depths} ----",
                          also_print=True)
    elif general_dict['experiment_type'] == "dense":
        denselayers = (32, 64, (64, 16), (32, 8), (16, 4))
        for d in denselayers:
            write_to_file(general_dict['write_file_path'], f"---- Starting Dense {d}/ {denselayers} ----",
                          also_print=True)
            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + f"_depth_{d}",
                                     add_dense=d)
            _ = model_object.fit(train_generator=train_generator,
                                 validation_generator=validation_generator,
                                 plot_test_acc=True,
                                 save_raw=False)

            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            write_to_file(general_dict['write_file_path'], f"---- Ending Depth {d}/ {denselayers} ----",
                          also_print=True)
    elif general_dict['experiment_type'] == "models":
        models = model_dict['model_name'].split(",")
        if "Net" in model_dict['model_name'] and not model_dict['use_conv2d']:
            raise ValueError("The models from eeg_models file needs 2D conv, dataset must be it as well!")

        for m in models:
            write_to_file(general_dict['write_file_path'], f"---- Starting Model {m}/ {models} ----",
                          also_print=True)
            model_object = get_model(which_model=m,
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=m)
            _ = model_object.fit(train_generator=train_generator,
                                 validation_generator=validation_generator,
                                 plot_test_acc=True,
                                 save_raw=False)

            eval_metrics = model_object.predict(data=test_generator, return_metrics=True)
            eval_metrics['majority_voting_acc'] = evaluate_majority_voting(model_object=model_object,
                                                                           test_dict=test_set_dictionary)

            write_to_file(general_dict['write_file_path'], f"Test Set Acc: {eval_metrics['accuracy']}"
                                                           f"\nMajority Voting Acc: "
                                                           f"{eval_metrics['majority_voting_acc']}", also_print=True)
            write_to_file(general_dict['write_file_path'], f"---- Ending Model {m}/ {models} ----",
                          also_print=True)

    elif general_dict['experiment_type'] == "temperature_scaling":

        model = keras.models.load_model("/home/tvetern/Dropbox/phd/projects/gender_prediction/keras_model",
                                        custom_objects={'mcc': mcc, 'specificity': specificity, 'recall': recall,
                                                        'precision': precision, 'f1': f1,
                                                        'auc': tf.keras.metrics.AUC})
