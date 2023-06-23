import configparser
import argparse
from myPack.utils import *
import shutil


def time_configparser(conf_file):
    """
    Function that receives a config file and extracts relevant information and returns it in dictionary form

    Args:
        conf_file: path to config file

    Returns:
        model_dict (dict): Dictionary with deep learning model specific parameters
        hyper_dict (dict): Dictionary with deep learning model training specific parameters
        general_dict (dict): Dictionary with general information, such as predictions, kfold, saving paths
    """
    config = configparser.ConfigParser()
    config.read(conf_file)

    model_dict = dict()
    hyper_dict = dict()
    time_dict = dict()
    general_dict = dict()

    model_dict['model_name'] = config.get('MODEL', 'model_name')
    model_dict['apply_mc'] = config.getboolean('MODEL', 'monte_carlo_dropout')
    model_dict['output_shape'] = 1

    if "inception" in model_dict['model_name']:
        model_dict['use_conv2d'] = False
    else:
        model_dict['use_conv2d'] = True

    hyper_dict['epochs'] = config.getint('HYPERPARAMETER', 'epochs')
    hyper_dict['patience'] = config.getint('HYPERPARAMETER', 'patience')
    hyper_dict['batch_size'] = config.getint('HYPERPARAMETER', 'batch_size')
    hyper_dict['lr'] = config.getfloat('HYPERPARAMETER', 'lr')
    hyper_dict['kernel_init'] = config.get('HYPERPARAMETER', 'kernel_init')
    hyper_dict['dropout'] = config.getfloat('HYPERPARAMETER', 'dropout')

    time_dict['srate'] = config.getint('TIME', 'sampling_rate')
    time_dict['num_windows'] = config.getint('TIME', 'num_windows')
    time_dict['num_datapoints_per_window'] = int(config.getfloat('TIME', 'num_seconds_per_window') * time_dict['srate'])
    time_dict['start_point'] = int(config.getfloat('TIME', 'starting_seconds') * time_dict['srate'])
    time_dict['train_set_every_other'] = config.getboolean('TIME', 'training_set_every_other_window')

    general_dict['save_path'] = config.get('GENERAL', 'result_save_path')
    general_dict['experiment_type'] = config.get('GENERAL', 'experiment_type')
    
    
    # Do multiple checks on the config file to make sure that the experiments is run correctly:
    if model_dict['apply_mc'] and general_dict['experiment_type'] != "single_model":
        raise ValueError("If Monte Carlo is used, it only works with experiment type: 'single_models'")
    elif general_dict['experiment_type'] == "ensemble_weights" and hyper_dict["kernel_init"] == "glorot_uniform":
        raise ValueError(f"Ensemble weights chose, but kernel init is set to {hyper_dict['kernel_init']}, "
                         f"should be random_normal or random_uniform")
    elif general_dict["experiment_type"] == "depth_ensemble" and model_dict["model_name"] != "inception":
        raise ValueError(f"Depth ensemble chosen, but with model: {model_dict['model_name']}, should be 'inception'!")
    elif general_dict['experiment_type'] == "ensemble_models" and len(model_dict['model_name'].split(",")) < 2:
        print("Ensemble Experiment chosen, but only one model is chosen, assumes that the model should be duplicated")
        model_dict['model_name'] = f"{model_dict['model_name']},{model_dict['model_name']}," \
                                   f"{model_dict['model_name']},{model_dict['model_name']}," \
                                   f"{model_dict['model_name']}"

    return model_dict, hyper_dict, time_dict, general_dict


def _argparse():
    """
    Function to handle command-line arguments:

    --conf refers to the config.ini file with all model and time-series parameters
    --dict path to the labels.plk file created with the dataset_creator.py script
    --test only use a few subjects to test that the code runs

    return the arguments
    """
    parser = argparse.ArgumentParser(description="Model Run")
    parser.add_argument('--conf', type=str, required=True,  help="Path to config")
    parser.add_argument('--dict', type=str, required=True, help='Path to .pkl file,labels')
    parser.add_argument('--test', default=False, action='store_true', help='Test with smaller dataset')

    return parser.parse_args()


def setup_run():
    """
    Function that sets up a run, with argument parser and config files
    """
    arg = _argparse()
    model_dict, hyper_dict, time_dict, general_dict = time_configparser(arg.conf)
    general_dict['testing'] = arg.test

    if general_dict['testing']:
        # If the test flag is active, reduce the number of epochs
        hyper_dict['epochs'] = 4

    # Creates a folder and copy the config file used in this run to the folder
    general_dict['save_path'], general_dict['fig_path'], general_dict['model_path'] = \
        create_run_folder(general_dict['save_path'])
    shutil.copyfile(arg.conf, general_dict['save_path'] + arg.conf)

    # Load the pkl file containig eeg file paths and all information about the subjects
    data_dict = load_pkl(arg.dict)

    # metrics and info is where the main metrics and information will be saved.
    general_dict['write_file_path'] = general_dict['save_path'] + "metrics_and_info.txt"
    f = open(general_dict['write_file_path'], "w")
    f.write(f"Conf: {arg.conf}\nDataset: {arg.dict}\n")
    f.close()

    return data_dict, model_dict, hyper_dict, time_dict, general_dict
