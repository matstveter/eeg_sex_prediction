from myPack.classifiers.inception_time import InceptionTime


def get_model(which_model: str, model_dict: dict, hyper_dict: dict, general_dict: dict, model_name):
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
        model_object = InceptionTime(**sim_args)
    elif which_model == "inception2":
        added_args = {'dense': 32}
        model_object = InceptionTime(**sim_args, **added_args)

    return model_object
