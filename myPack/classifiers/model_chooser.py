from myPack.classifiers.eeg_models import EEGnet, EEGnet_own, EEGnet_own2, EEGnet_own4
from myPack.classifiers.inception_time import InceptionTime, InceptionTime2
from myPack.classifiers.time_classifiers import ExperimentalClassifier


def get_model(which_model: str, model_dict: dict, hyper_dict: dict, general_dict: dict, model_name, **kwargs):
    sim_args = {'input_shape': model_dict['input_shape'],
                'output_shape': 1,
                'save_path': general_dict['model_path'],
                'fig_path': general_dict['fig_path'],
                'save_name': model_name,
                'logits': model_dict['model_with_logits'],
                'batch_size': hyper_dict['batch_size'],
                'epochs': hyper_dict['epochs'],
                'patience': hyper_dict['patience'],
                'learning_rate': hyper_dict['lr'],
                'verbose': False}

    if which_model == "inception":
        model_object = InceptionTime(**sim_args, **kwargs)

    elif which_model == "eegNet":
        model_object = EEGnet(**sim_args, **kwargs)

    elif which_model == "eegNet2":
        model_object = EEGnet_own(**sim_args, **kwargs)

    elif which_model == "eegNet3":
        model_object = EEGnet_own2(**sim_args, **kwargs)

    elif which_model == "eegNet4":
        model_object = EEGnet_own4(**sim_args, **kwargs)

    elif which_model == "inception2":
        model_object = InceptionTime2(**sim_args, **kwargs)

    elif which_model == "experimental":
        model_object = ExperimentalClassifier(**sim_args, **kwargs)

    else:
        raise ValueError(f"Can not recognize model name: {which_model}")

    return model_object
