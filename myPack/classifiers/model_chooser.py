from myPack.classifiers.eeg_models import EEGnet, EEGnet_own
from myPack.classifiers.inception_time import InceptionTime


def get_model(which_model: str, model_dict: dict, hyper_dict: dict, general_dict: dict, model_name, depth=None):
    sim_args = {'input_shape': model_dict['input_shape'],
                'output_shape': 1,
                'save_path': general_dict['model_path'],
                'fig_path': general_dict['fig_path'],
                'save_name': which_model + "_" + model_name,
                'logits': False,
                'batch_size': hyper_dict['batch_size'],
                'epochs': hyper_dict['epochs'],
                'patience': hyper_dict['patience'],
                'learning_rate': hyper_dict['lr'],
                'verbose': False,
                'kernel_init': hyper_dict['kernel_init']}

    if which_model == "inception":
        if depth is None:
            model_object = InceptionTime(**sim_args, depth=4)
        else:
            model_object = InceptionTime(**sim_args, depth=depth)
    elif which_model == "inception_mc":
        model_object = InceptionTime(**sim_args, depth=4, add_dense=32)
    elif which_model == "inception_deeper":
        model_object = InceptionTime(**sim_args, depth=8)
    elif which_model == "eegNet":
        model_object = EEGnet(**sim_args)
    elif which_model == "eegNet_MC":
        model_object = EEGnet(**sim_args, add_dense=32)
    elif which_model == "eegNet_less_drop":
        model_object = EEGnet(**sim_args, dropout_rate=0.25)
    elif which_model == "eegNet_deeper":
        model_object = EEGnet_own(**sim_args)
    else:
        raise ValueError(f"Can not recognize model name: {which_model}")

    return model_object
