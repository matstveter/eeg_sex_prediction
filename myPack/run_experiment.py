import keras.backend
import os
import gc
import tensorflow as tf
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from myPack.classifiers.eeg_models import DeepConvNet, EEGnet, EEGnet_SSVEP, ShallowNet
from myPack.classifiers.inception_time import InceptionTime
from myPack.classifiers.model_with_temperature import ModelTemp, ModelWithTemperature
from myPack.classifiers.time_classifiers import TimeClassifer, ExperimentalClassifier
from myPack.data.data_handler import DataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# from myPack.data.data_handling import create_split_from_dict, load_time_series_dataset
from myPack.utils import save_to_pkl, load_pkl
from myPack.utils import write_to_file
from myPack.eval.ensemble_eval import evaluate_ensemble_majority

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
    elif model_name == "deepconvnet":
        if model_dict['input_shape'][-1] == 1:
            model_object = DeepConvNet(input_shape=model_dict['input_shape'],
                                       output_shape=1,
                                       save_path=general_dict['model_path'],
                                       fig_path=general_dict['fig_path'],
                                       save_name=name_of_model,
                                       logits=model_dict['model_with_logits'],
                                       batch_size=hyper_dict['batch_size'],
                                       epochs=hyper_dict['epochs'],
                                       patience=hyper_dict['patience'],
                                       learning_rate=hyper_dict['lr'])
        else:
            raise ValueError("DeepConvNet expectes 4 dimensional input, because of conv2d!")
    elif model_name == "shallownet":
        if model_dict['input_shape'][-1] == 1:
            model_object = ShallowNet(input_shape=model_dict['input_shape'],
                                      output_shape=1,
                                      save_path=general_dict['model_path'],
                                      fig_path=general_dict['fig_path'],
                                      save_name=name_of_model,
                                      logits=model_dict['model_with_logits'],
                                      batch_size=hyper_dict['batch_size'],
                                      epochs=hyper_dict['epochs'],
                                      patience=hyper_dict['patience'],
                                      learning_rate=hyper_dict['lr'])
        else:
            raise ValueError("ShallowNet expectes 4 dimensional input, because of conv2d!")
    elif model_name == "eegnet":
        if model_dict['input_shape'][-1] == 1:
            model_object = EEGnet(input_shape=model_dict['input_shape'],
                                  output_shape=1,
                                  save_path=general_dict['model_path'],
                                  fig_path=general_dict['fig_path'],
                                  save_name=name_of_model,
                                  logits=model_dict['model_with_logits'],
                                  batch_size=hyper_dict['batch_size'],
                                  epochs=hyper_dict['epochs'],
                                  patience=hyper_dict['patience'],
                                  learning_rate=hyper_dict['lr'])
        else:
            raise ValueError("EEGnet expectes 4 dimensional input, because of conv2d!")
    elif model_name == "eegnet_ssvep":
        if model_dict['input_shape'][-1] == 1:
            model_object = EEGnet_SSVEP(input_shape=model_dict['input_shape'],
                                        output_shape=1,
                                        save_path=general_dict['model_path'],
                                        fig_path=general_dict['fig_path'],
                                        save_name=name_of_model,
                                        logits=model_dict['model_with_logits'],
                                        batch_size=hyper_dict['batch_size'],
                                        epochs=hyper_dict['epochs'],
                                        patience=hyper_dict['patience'],
                                        learning_rate=hyper_dict['lr'])
        else:
            raise ValueError("EEGnet_SSVEP expectes 4 dimensional input, because of conv2d!")
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
    elif model_name == "exp":
        model_object = ExperimentalClassifier(input_shape=model_dict['input_shape'],
                                              output_shape=1,
                                              save_path=general_dict['model_path'],
                                              fig_path=general_dict['fig_path'],
                                              save_name=name_of_model,
                                              logits=model_dict['model_with_logits'],
                                              batch_size=hyper_dict['batch_size'],
                                              epochs=hyper_dict['epochs'],
                                              patience=hyper_dict['patience'],
                                              learning_rate=hyper_dict['lr'])
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


def run_experiments(data_dict, model_dict, hyper_dict, time_dict, general_dict):
    NUM_MODELS = 5


    if general_dict['experiment_type'] == "run":
        train_id, val_id, test_id, large_dataset = create_split_from_dict(data_dict)

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

            # model_object.model.save("keras_model")

            # temp_model = ModelWithTemperature(model=model_object.model, batch_size=hyper_dict['batch_size'],
            #                                   save_path=general_dict['fig_path'] + "model_" + str(i))
            # temp_model.set_temperature(valid_loader=val_gen)

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

            keras.backend.clear_session()
            _ = gc.collect()
            write_to_file(general_dict['write_file_path'], f"************* Ending Model Run: {i} *************\n\n",
                          also_print=True)
        save_to_pkl(result_dict, general_dict['save_path'], "result_dictionary")
    elif general_dict['experiment_type'] == "temp":
        from myPack.classifiers.keras_utils import mcc, specificity, recall, precision, f1

        train_id, val_id, test_id, large_dataset = create_split_from_dict(data_dict)
        if general_dict['testing']:
            # val_id = val_id[:10]
            pass

        train_gen, val_gen, model_shape, test_x, test_y, test_dict = get_generators(train_id=train_id[0:1],
                                                                                    val_id=val_id,
                                                                                    test_id=test_id,
                                                                                    data_dict=data_dict,
                                                                                    hyper_dict=hyper_dict,
                                                                                    time_dict=time_dict,
                                                                                    large_data=large_dataset,
                                                                                    conv2d=model_dict['use_conv2d'])
        model_dict['input_shape'] = model_shape

        model = keras.models.load_model("/home/tvetern/Dropbox/phd/projects/gender_prediction/keras_model",
                                        custom_objects={'mcc': mcc, 'specificity': specificity, 'recall': recall,
                                                        'precision': precision, 'f1': f1,
                                                        'auc': tf.keras.metrics.AUC})
        # temp_model = ModelWithTemperature(model=model, batch_size=hyper_dict['batch_size'],
        #                                   save_path=general_dict['fig_path'], opt="sgd")
        # temp_model.set_temperature(valid_loader=val_gen)

        # temp2 = ModelWithTemperature(model=model, batch_size=hyper_dict['batch_size'],
        #                              save_path=general_dict['fig_path'], opt="adam")
        # temp2.set_temperature(valid_loader=val_gen)

        # temp3 = ModelWithTemperature(model=model, batch_size=hyper_dict['batch_size'],
        #                              save_path=general_dict['fig_path'], opt="ftrl")
        # temp3.set_temperature(valid_loader=val_gen)

        # temp4 = ModelWithTemperature(model=model, batch_size=hyper_dict['batch_size'],
        #                              save_path=general_dict['fig_path'], opt="rms")
        # temp4.set_temperature(valid_loader=val_gen)

        temp = ModelTemp(model=model)
        _ = temp.set_temperature(valid_loader=val_gen)

    else:
        print(f"Unrecognized experiment type: {general_dict['experiment_type']}")
