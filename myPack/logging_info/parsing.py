import configparser
import argparse
from myPack.utils import *
import shutil


def get_info_from_config(path, section, config_key):
    """
    Function which returns a specific value from the config file

    Args:
        section: which section
        path: path to config-file
        config_key: what part of the config file should be collected

    Returns:
    The data specified in the config_key
    """
    config = configparser.ConfigParser()
    config.read(path)

    return config.get(section, config_key)


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
    model_dict['use_conv2d'] = config.getboolean('MODEL', 'conv_2d')

    hyper_dict['epochs'] = config.getint('HYPERPARAMETER', 'epochs')
    hyper_dict['patience'] = config.getint('HYPERPARAMETER', 'patience')
    hyper_dict['batch_size'] = config.getint('HYPERPARAMETER', 'batch_size')
    hyper_dict['lr'] = config.getfloat('HYPERPARAMETER', 'lr')
    hyper_dict['kernel_init'] = config.get('HYPERPARAMETER', 'kernel_init')
    hyper_dict['dropout'] = config.getfloat('HYPERPARAMETER', 'dropout')
    hyper_dict['cnn_dropout'] = config.getfloat('HYPERPARAMETER', 'cnn_dropout')

    time_dict['srate'] = config.getint('TIME', 'sampling_rate')
    time_dict['num_windows'] = config.getint('TIME', 'num_windows')
    time_dict['num_datapoints_per_window'] = int(config.getfloat('TIME', 'num_seconds_per_window') * time_dict['srate'])
    time_dict['start_point'] = int(config.getfloat('TIME', 'starting_seconds') * time_dict['srate'])
    time_dict['train_set_every_other'] = config.getboolean('TIME', 'training_set_every_other_window')

    general_dict['save_path'] = config.get('GENERAL', 'save_path')
    general_dict['prediction'] = "Sex"
    general_dict['experiment_type'] = config.get('GENERAL', 'experiment_type')

    # todo Check that if the chosen model is InceptionTime, then use_conv2d must be false

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
        hyper_dict['epochs'] = 3

    general_dict['save_path'], general_dict['fig_path'], general_dict['model_path'] = \
        create_run_folder(general_dict['save_path'])
    shutil.copyfile(arg.conf, general_dict['save_path'] + arg.conf)
    data_dict = load_pkl(arg.dict)

    return data_dict, model_dict, hyper_dict, time_dict, general_dict
