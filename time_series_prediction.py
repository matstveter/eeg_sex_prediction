from myPack.classifiers.model_chooser import get_model
from myPack.final_run import run_final_experiment
from myPack.run_experiment import run_experiments
from myPack.logging_info.parsing import setup_run
from myPack.data.data_handler import get_all_data_and_generators


def main():
    data_dict, model_dict, hyper_dict, time_dict, general_dict = setup_run()
    print(general_dict['test_mode'])

    if general_dict['test_mode']:
        pass
    else:
        run_final_experiment(data_dict=data_dict, model_dict=model_dict, hyper_dict=hyper_dict,
                             time_dict=time_dict, general_dict=general_dict)


if __name__ == "__main__":
    main()