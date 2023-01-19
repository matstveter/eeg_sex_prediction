from myPack.classifiers.eeg_models import ShallowNet, DeepConvNet, EEGnet, EEGnet_SSVEP
from myPack.classifiers.inception_time import InceptionTime


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
    elif which_model == "shallowNet":
        model_object = ShallowNet(**sim_args, **kwargs)
    elif which_model == "deepConvNet":
        model_object = DeepConvNet(**sim_args, **kwargs)
    elif which_model == "eegNet":
        model_object = EEGnet(**sim_args, **kwargs)
    elif which_model == "eegNetS":
        model_object = EEGnet_SSVEP(**sim_args, **kwargs)
    else:
        raise ValueError(f"Can not recognize model name: {which_model}")

    return model_object
