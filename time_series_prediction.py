from myPack.run_experiment import run_experiments
from myPack.logging_info.parsing import setup_run


def main():
    data_dict, model_dict, hyper_dict, general_dict = setup_run()

    if general_dict['test_mode']:
        run_tests()
    else:
        run_experiments()


if __name__ == "__main__":
    data_dict, model_dict, hyper_dict, time_dict, general_dict = setup_run()
    run_experiments(data_dict=data_dict,
                    model_dict=model_dict,
                    hyper_dict=hyper_dict,
                    time_dict=time_dict,
                    general_dict=general_dict)
