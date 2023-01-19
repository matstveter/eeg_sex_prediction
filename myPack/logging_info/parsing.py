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
    model_dict['model_with_logits'] = config.getboolean('MODEL', 'model_with_logits')
    model_dict['apply_mc'] = config.getboolean('MODEL', 'monte_carlo_dropout')
    model_dict['output_shape'] = 1
    model_dict['use_conv2d'] = config.getboolean('MODEL', 'conv2d')

    hyper_dict['epochs'] = config.getint('HYPERPARAMETER', 'epochs')
    hyper_dict['patience'] = config.getint('HYPERPARAMETER', 'patience')
    hyper_dict['batch_size'] = config.getint('HYPERPARAMETER', 'batch_size')
    hyper_dict['lr'] = config.getfloat('HYPERPARAMETER', 'lr')
    hyper_dict['kernel_init'] = config.get('HYPERPARAMETER', 'kernel_init')

    time_dict['srate'] = config.getint('TIME', 'sampling_rate')
    time_dict['num_windows'] = config.getint('TIME', 'num_windows')
    time_dict['num_datapoints_per_window'] = int(config.getfloat('TIME', 'num_seconds_per_window') * time_dict['srate'])
    time_dict['start_point'] = int(config.getfloat('TIME', 'starting_seconds') * time_dict['srate'])
    time_dict['train_set_every_other'] = config.getboolean('TIME', 'training_set_every_other_window')

    general_dict['test_mode'] = config.getboolean('GENERAL', 'test_mode')
    general_dict['experiment_type'] = config.get('GENERAL', 'experiment_type')
    general_dict['num_models'] = config.getint('GENERAL', 'num_models')

    hyper_dict['dropout'] = 0.5
    hyper_dict['cnn_dropout'] = 0.1
    general_dict['save_path'] = "/home/tvetern/datasets/Results/"

    return model_dict, hyper_dict, time_dict, general_dict


def _argparse():
    """
    Function to handle input arguments from terminal window
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

    if arg.test:
        hyper_dict['epochs'] = 4

    general_dict['save_path'], general_dict['fig_path'], general_dict['model_path'] = \
        create_run_folder(general_dict['save_path'])
    shutil.copyfile(arg.conf, general_dict['save_path'] + arg.conf)
    data_dict = load_pkl(arg.dict)

    general_dict['write_file_path'] = general_dict['save_path'] + "metrics_and_info.txt"
    f = open(general_dict['write_file_path'], "w")
    f.close()

    return data_dict, model_dict, hyper_dict, time_dict, general_dict
