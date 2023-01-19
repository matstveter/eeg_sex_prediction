import numpy as np

from myPack.classifiers.model_chooser import get_model
from myPack.data.data_handler import get_all_data_and_generators
from myPack.utils import write_to_file


def run_final_experiment(data_dict: dict, model_dict: dict, hyper_dict: dict, time_dict: dict, general_dict: dict):

    # Get data generators, test_dictionary and model shape
    train_generator, validation_generator, test_generator, test_set_dictionary, model_shape = \
        get_all_data_and_generators(data_dict=data_dict, time_dict=time_dict, batch_size=hyper_dict['batch_size'],
                                    use_conv2d=model_dict['use_conv2d'], test=general_dict['testing'])
    model_dict['input_shape'] = model_shape

    # Write to file, important information
    write_to_file(general_dict['write_file_path'], message=f"**** Starting Experiment: "
                                                           f"{general_dict['experiment_type']} ****"
                                                           f"\nModel(s): {model_dict['model_name']}", also_print=False)

    if general_dict['experiment_type'] == "single_model":
        # Depending on the model that is running, this can adjust the models "extra" parameters
        added_args = {}

        for i in range(general_dict['num_models']):

            model_object = get_model(which_model=model_dict['model_name'],
                                     model_dict=model_dict,
                                     hyper_dict=hyper_dict,
                                     general_dict=general_dict,
                                     model_name=model_dict['model_name'] + "_" + str(i),
                                     **added_args)
            print(model_object.predict(data=test_generator, return_metrics=True))

            a = np.zeros((10, 129, 1000))
            b = np.zeros((10))
            print(model_object.predict(data=a, labels=b, return_metrics=True))
    elif general_dict['experiment_type'] == "ensemble_models":
        pass
    elif general_dict['experiment_type'] == "ensemble_weights":
        pass
    else:
        raise ValueError(f"Experiment type is not recognized : {general_dict['experiment_type']}!")
