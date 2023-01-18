import keras.backend
import os
import gc
import tensorflow as tf
import numpy as np

from myPack.classifiers.inception_time import InceptionTime, InceptionTime2, InceptionTime3
from myPack.classifiers.model_with_temperature import ModelWithTemperature
from myPack.classifiers.time_classifiers import TimeClassifer
from myPack.data.data_generator import DataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from myPack.data.data_handling import create_split_from_dict, load_time_series_dataset, get_all_data
from myPack.utils import save_to_pkl, load_pkl
from myPack.utils import write_to_file
from myPack.uncertainty_explainability.ensemble_eval import evaluate_ensemble_majority

config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
keras.backend.set_session(session)


def choose_model(model_name, model_dict: dict, general_dict: dict, name_of_model: str, hyper_dict: dict, depth=4,
                 kernels=(10, 20, 40)):
    if model_name == "time":
        model_object = TimeClassifer(input_shape=model_dict['input_shape'],
                                     output_shape=1,
                                     save_path=general_dict['model_path'],
                                     fig_path=general_dict['fig_path'],
                                     save_name=name_of_model,
                                     logits=model_dict['model_with_logits'],
                                     batch_size=hyper_dict['batch_size'],
                                     epochs=hyper_dict['epochs'],
                                     patience=hyper_dict['patience'],
                                     learning_rate=hyper_dict['lr'])
    elif model_name == "inc":
        model_object = InceptionTime(input_shape=model_dict['input_shape'],
                                     output_shape=1,
                                     save_path=general_dict['model_path'],
                                     fig_path=general_dict['fig_path'],
                                     save_name=name_of_model,
                                     logits=model_dict['model_with_logits'],
                                     batch_size=hyper_dict['batch_size'],
                                     epochs=hyper_dict['epochs'],
                                     patience=hyper_dict['patience'],
                                     learning_rate=hyper_dict['lr'])
    elif model_name == "inc2":
        model_object = InceptionTime2(input_shape=model_dict['input_shape'],
                                      output_shape=1,
                                      save_path=general_dict['model_path'],
                                      fig_path=general_dict['fig_path'],
                                      save_name=name_of_model,
                                      logits=model_dict['model_with_logits'],
                                      batch_size=hyper_dict['batch_size'],
                                      epochs=hyper_dict['epochs'],
                                      patience=hyper_dict['patience'],
                                      learning_rate=hyper_dict['lr'],
                                      depth=depth,
                                      kernel_sizes=kernels)
    elif model_name == "inc3":
        model_object = InceptionTime3(input_shape=model_dict['input_shape'],
                                      output_shape=1,
                                      save_path=general_dict['model_path'],
                                      fig_path=general_dict['fig_path'],
                                      save_name=name_of_model,
                                      logits=model_dict['model_with_logits'],
                                      batch_size=hyper_dict['batch_size'],
                                      epochs=hyper_dict['epochs'],
                                      patience=hyper_dict['patience'],
                                      learning_rate=hyper_dict['lr'],
                                      depth=depth,
                                      kernel_sizes=kernels)
    else:
        raise ValueError("Unrecognizable model_name!")

    return model_object


def evaluate_majority(model_object, test_dict: dict, write_file: str) -> float:
    """ Evaluates the model using majority voting

    Args:
        model_object: a type of classifier object
        test_dict:  dictionary containing, the test data, and labels
        write_file: path to the file where the results is written

    Returns:
        float: majority_voting acc
    """
    num_correct_subjects_majority_voting = 0

    for subject, value in test_dict.items():
        data_x = np.array(value['data'])
        data_y = np.array([value['label']] * data_x.shape[0])
        acc, _, num_correct = model_object.predict(x_test=data_x, y_test=data_y, return_metrics=True)

        if acc > 0.5:
            num_correct_subjects_majority_voting += 1

    majority_acc = (num_correct_subjects_majority_voting / len(test_dict.keys()))
    write_to_file(write_file, f"Majority Voting: {majority_acc}")
    return majority_acc


def get_generators(train_id: list, val_id: list, test_id: list, data_dict: dict, hyper_dict: dict, time_dict: dict,
                   large_data=False, conv2d=True):
    training_generator = DataGenerator(list_IDs=train_id,
                                       data_dict=data_dict,
                                       time_slice=time_dict['num_datapoints_per_window'],
                                       train_set=time_dict['train_set_every_other'],
                                       batch_size=hyper_dict['batch_size'],
                                       start=time_dict['start_point'],
                                       num_windows=time_dict['num_windows'],
                                       large_data=large_data,
                                       conv2d=conv2d)

    val_generator = DataGenerator(list_IDs=val_id,
                                  data_dict=data_dict,
                                  time_slice=time_dict['num_datapoints_per_window'],
                                  train_set=False,
                                  batch_size=hyper_dict['batch_size'],
                                  start=time_dict['start_point'],
                                  num_windows=time_dict['num_windows'],
                                  conv2d=conv2d)

    print("Training Generator Length: ", len(training_generator))
    print("Validation Generator Length: ", len(val_generator))

    test_x, test_y, test_dict = load_time_series_dataset(participant_list=test_id[0:2],
                                                         data_dict=data_dict,
                                                         datapoints_per_window=time_dict[
                                                             'num_datapoints_per_window'],
                                                         number_of_windows=time_dict['num_windows'] * 2,
                                                         starting_point=time_dict['start_point'],
                                                         conv2d=conv2d,
                                                         train_set=False)
    model_shape = test_x.shape[1:]
    print(f"Input data shape is: {model_shape}")

    return training_generator, val_generator, model_shape, test_x, test_y, test_dict


def run_experiments(data_dict, model_dict, hyper_dict, time_dict, general_dict):
    NUM_MODELS = 5

    general_dict['write_file_path'] = general_dict['save_path'] + "metrics_and_info.txt"
    f = open(general_dict['write_file_path'], "w")
    f.close()

    if general_dict['experiment_type'] == "single":
        train_x, train_y, val_x, val_y, test_x, test_y, test_dict, _ = \
            get_all_data(data_dict=data_dict, general_dict=general_dict, time_dict=time_dict,
                         conv2d=model_dict['use_conv2d'])
        model_dict['input_shape'] = train_x.shape[1:]

        for i in range(NUM_MODELS):
            write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {i} *************",
                          also_print=True)

            model_object = choose_model(model_name=model_dict['model_name'], model_dict=model_dict,
                                        general_dict=general_dict, name_of_model="model_" + str(i),
                                        hyper_dict=hyper_dict)

            model_object.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, save_raw=True,
                             plot_test_acc=True)

            acc, report, num_correct = model_object.predict(x_test=test_x, y_test=test_y, return_metrics=True)

            evaluate_majority(model_object=model_object, test_dict=test_dict,
                              write_file=general_dict['write_file_path'])
            write_to_file(general_dict['write_file_path'], f"Test Set Performance acc: {acc}")

            _ = gc.collect()
            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                          also_print=True)
    elif general_dict['experiment_type'] == "generator":
        if len(list(data_dict.keys())) > 2000:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.7, 0.2, 0.1])
            large_dataset = True
        else:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.6, 0.2, 0.2])
            large_dataset = False

        if general_dict['testing']:
            # train_id = train_id[0:10]
            # val_id = val_id[0:5]
            # test_id = test_id[0:5]
            NUM_MODELS = 2

        print(f"Train ID: {len(train_id)}, Val ID: {len(val_id)}, Test ID: {len(test_id)}")

        train_gen, val_gen, model_shape, test_x, test_y, test_dict = get_generators(train_id=train_id,
                                                                                    val_id=val_id,
                                                                                    test_id=test_id,
                                                                                    data_dict=data_dict,
                                                                                    hyper_dict=hyper_dict,
                                                                                    time_dict=time_dict,
                                                                                    large_data=large_dataset,
                                                                                    conv2d=model_dict['use_conv2d'])

        model_dict['input_shape'] = model_shape

        result_dict = dict()
        for i in range(NUM_MODELS):
            write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {i} *************",
                          also_print=True)
            model_object = choose_model(model_name=model_dict['model_name'], model_dict=model_dict,
                                        general_dict=general_dict, name_of_model="model_" + str(i),
                                        hyper_dict=hyper_dict)

            model_object.fit(train_x=train_gen, train_y=None, val_x=val_gen, val_y=None, save_raw=True,
                             plot_test_acc=True)

            model_object.model.save("keras_model")

            temp_model = ModelWithTemperature(model=model_object.model, batch_size=hyper_dict['batch_size'],
                                              save_path=general_dict['fig_path'] + "model_" + str(i))
            temp_model.set_temperature(valid_loader=val_gen)

            print("[INFO] Trying to predict test set")
            # train_gen = None
            # val_gen = None
            test_x, test_y, test_dict = load_time_series_dataset(participant_list=test_id,
                                                                 data_dict=data_dict,
                                                                 datapoints_per_window=time_dict[
                                                                     'num_datapoints_per_window'],
                                                                 number_of_windows=40,
                                                                 starting_point=time_dict['start_point'],
                                                                 conv2d=model_dict['use_conv2d'],
                                                                 train_set=False)

            acc, report, num_correct = model_object.predict(x_test=test_x, y_test=test_y, return_metrics=True)
            write_to_file(general_dict['write_file_path'], f"Test Set Performance acc: {acc}")

            evaluate_majority(model_object=model_object, test_dict=test_dict,
                              write_file=general_dict['write_file_path'])

            _ = gc.collect()
            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                          also_print=True)
        save_to_pkl(result_dict, general_dict['save_path'], "result_dictionary")
    elif general_dict['experiment_type'] == "inc_test_kernels":
        if len(list(data_dict.keys())) > 2000:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.7, 0.2, 0.1])
            large_dataset = True
        else:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.6, 0.2, 0.2])
            large_dataset = False

        print(f"Train ID: {len(train_id)}, Val ID: {len(val_id)}, Test ID: {len(test_id)}")

        train_gen, val_gen, model_shape, test_x, test_y, test_dict = get_generators(train_id=train_id,
                                                                                    val_id=val_id,
                                                                                    test_id=test_id,
                                                                                    data_dict=data_dict,
                                                                                    hyper_dict=hyper_dict,
                                                                                    time_dict=time_dict,
                                                                                    large_data=large_dataset,
                                                                                    conv2d=model_dict['use_conv2d'])
        model_dict['input_shape'] = model_shape

        kernels = ((5, 10, 20), (7, 15, 30), (5, 20, 60))

        for k in kernels:
            write_to_file(general_dict['write_file_path'], f"************* Starting kernel: {k} *************",
                          also_print=True)
            for i in range(3):
                write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {i} *************",
                              also_print=True)
                model_object = choose_model(model_name=model_dict['model_name'], model_dict=model_dict,
                                            general_dict=general_dict, name_of_model="model_" + str(i),
                                            hyper_dict=hyper_dict, kernels=k)

                model_object.fit(train_x=train_gen, train_y=None, val_x=val_gen, val_y=None, save_raw=True,
                                 plot_test_acc=True)

                train_gen = None
                val_gen = None
                test_x, test_y, test_dict = load_time_series_dataset(participant_list=test_id,
                                                                     data_dict=data_dict,
                                                                     datapoints_per_window=time_dict[
                                                                         'num_datapoints_per_window'],
                                                                     number_of_windows=40,
                                                                     starting_point=time_dict['start_point'],
                                                                     conv2d=model_dict['use_conv2d'],
                                                                     train_set=False)

                acc, report, num_correct = model_object.predict(x_test=test_x, y_test=test_y, return_metrics=True)
                write_to_file(general_dict['write_file_path'], f"Test Set Performance acc: {acc}")

                evaluate_majority(model_object=model_object, test_dict=test_dict,
                                  write_file=general_dict['write_file_path'])

                _ = gc.collect()
                write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                              also_print=True)
                break
            write_to_file(general_dict['write_file_path'], f"************* Ending kernel: {k} *************\n\n",
                          also_print=True)
    elif general_dict['experiment_type'] == "inc_test_depth":
        if len(list(data_dict.keys())) > 2000:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.7, 0.2, 0.1])
            large_dataset = True
        else:
            train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                               split=[0.6, 0.2, 0.2])
            large_dataset = False

        print(f"Train ID: {len(train_id)}, Val ID: {len(val_id)}, Test ID: {len(test_id)}")

        train_gen, val_gen, model_shape, test_x, test_y, test_dict = get_generators(train_id=train_id,
                                                                                    val_id=val_id,
                                                                                    test_id=test_id,
                                                                                    data_dict=data_dict,
                                                                                    hyper_dict=hyper_dict,
                                                                                    time_dict=time_dict,
                                                                                    large_data=large_dataset,
                                                                                    conv2d=model_dict['use_conv2d'])
        model_dict['input_shape'] = model_shape

        depths = (2, 4, 6, 8, 10)

        for d in depths:
            write_to_file(general_dict['write_file_path'], f"************* Starting deptjh: {d} *************",
                          also_print=True)
            for i in range(3):
                write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {i} *************",
                              also_print=True)
                model_object = choose_model(model_name=model_dict['model_name'], model_dict=model_dict,
                                            general_dict=general_dict, name_of_model="model_" + str(i),
                                            hyper_dict=hyper_dict, depth=d)

                model_object.fit(train_x=train_gen, train_y=None, val_x=val_gen, val_y=None, save_raw=True,
                                 plot_test_acc=True)

                train_gen = None
                val_gen = None
                test_x, test_y, test_dict = load_time_series_dataset(participant_list=test_id,
                                                                     data_dict=data_dict,
                                                                     datapoints_per_window=time_dict[
                                                                         'num_datapoints_per_window'],
                                                                     number_of_windows=40,
                                                                     starting_point=time_dict['start_point'],
                                                                     conv2d=model_dict['use_conv2d'],
                                                                     train_set=False)
                acc, report, num_correct = model_object.predict(x_test=test_x, y_test=test_y, return_metrics=True)
                write_to_file(general_dict['write_file_path'], f"Test Set Performance acc: {acc}")

                evaluate_majority(model_object=model_object, test_dict=test_dict,
                                  write_file=general_dict['write_file_path'])

                _ = gc.collect()
                write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                              also_print=True)
                break
            write_to_file(general_dict['write_file_path'], f"************* Ending depth: {d} *************\n\n",
                          also_print=True)
    elif general_dict['experiment_type'] == "ensemble_weights":
        train_x, train_y, val_x, val_y, test_x, test_y, test_dict, _ = \
            get_all_data(data_dict=data_dict, general_dict=general_dict, time_dict=time_dict,
                         conv2d=model_dict['use_conv2d'])

        result_dict = dict()
        models = list()
        for i in range(NUM_MODELS):
            write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {i} *************",
                          also_print=True)
            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                          also_print=True)

        resu_dict = evaluate_ensemble_majority(model=models, test_dict=test_dict,
                                               write_file=general_dict['write_file_path'])
        # Save all models, or the best models
        result_dict['ensemble'] = resu_dict
        save_to_pkl(result_dict, general_dict['save_path'], "result_dictionary")

    elif general_dict['experiment_type'] == "ensemble_models":
        train_x, train_y, val_x, val_y, test_x, test_y, test_dict, _ = \
            get_all_data(data_dict=data_dict, general_dict=general_dict, time_dict=time_dict,
                         conv2d=model_dict['use_conv2d'])
        ensemble_models = []
        models = list()
        result_dict = dict()

        for e in ensemble_models:
            write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {e} *************",
                          also_print=True)

            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {e} *************\n\n",
                          also_print=True)

        resu_dict = evaluate_ensemble_majority(model=models, test_dict=test_dict,
                                               write_file=general_dict['write_file_path'])

        result_dict['ensemble'] = resu_dict
        save_to_pkl(result_dict, general_dict['save_path'], "result_dictionary")

    elif general_dict['experiment_type'] == "ensemble_freq":

        ensemble_models = model_dict['model_name'].split(',')
        specific_models = False

        if len(ensemble_models) == 1:
            ensemble_models = ensemble_models * 5
        else:
            specific_models = True

        path = "/home/tvetern/datasets/numpy/"
        path_2 = "/media/hdd/created_datasets/"
        path_extensions = ["normal_delta", "normal_theta", "normal_alpha", "normal_beta", "normal_gamma"]
        final_dataset_path = path + "prep_no_change/"
        paths = [path_2 + path_extensions[i] + "/" for i in range(len(path_extensions))]

        models = list()
        result_dict = dict()

        for count, p in enumerate(paths):
            data_dict = load_pkl(p)
            train_x, train_y, val_x, val_y, test_x, test_y, test_dict, _ = \
                get_all_data(data_dict=data_dict, general_dict=general_dict, time_dict=time_dict,
                             conv2d=model_dict['use_conv2d'])

            write_to_file(general_dict['write_file_path'], f"************* Starting Model Run: {count} *************",
                          also_print=True)

            if specific_models:
                model_dict['model_name'] = str(p.split("_")[2][:-1])
                print(model_dict['model_name'])
            else:
                model_dict['model_name'] = ensemble_models[count]

            write_to_file(general_dict['write_file_path'], f"Running model: {model_dict['model_name']}\n",
                          also_print=True)

            # my_model.save_model(which_model='loaded_weights')
            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {count} *************\n\n",
                          also_print=True)
            train_x = None
            train_y = None
            val_x = None
            val_y = None
            test_x = None
            test_y = None
            _ = gc.collect()
            keras.backend.clear_session()

        data_dict = load_pkl(final_dataset_path)

        train_x, train_y, val_x, val_y, test_x, test_y, test_dict, _ = \
            get_all_data(data_dict=data_dict, general_dict=general_dict, time_dict=time_dict,
                         conv2d=model_dict['use_conv2d'])

        resu_dict = evaluate_ensemble_majority(model=models, test_dict=test_dict,
                                               write_file=general_dict['write_file_path'])

        result_dict['ensemble'] = resu_dict
        save_to_pkl(result_dict, general_dict['save_path'], "result_dictionary")
    elif general_dict['experiment_type'] == "temp":

        train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'],
                                                           split=[0.7, 0.2, 0.1])
        large_dataset = True
        train_gen, val_gen, model_shape, test_x, test_y, test_dict = get_generators(train_id=train_id[0:1],
                                                                                    val_id=val_id,
                                                                                    test_id=test_id,
                                                                                    data_dict=data_dict,
                                                                                    hyper_dict=hyper_dict,
                                                                                    time_dict=time_dict,
                                                                                    large_data=large_dataset,
                                                                                    conv2d=model_dict['use_conv2d'])
        model_dict['input_shape'] = model_shape
        
        model = keras.models.load_model("/home/tvetern/Dropbox/phd/projects/gender_prediction/")
        temp_model = ModelWithTemperature(model=model, batch_size=hyper_dict['batch_size'],
                                          save_path=general_dict['fig_path'] + "model_")
        temp_model.set_temperature(valid_loader=val_gen)

    else:
        print(f"Unrecognized experiment type: {general_dict['experiment_type']}")
