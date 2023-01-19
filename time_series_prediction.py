from myPack.final_run import run_final_experiment
from myPack.logging_info.parsing import setup_run
from myPack.tester import testing_models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def main():
    data_dict, model_dict, hyper_dict, time_dict, general_dict = setup_run()

    if general_dict['test_mode']:
        testing_models(data_dict=data_dict, model_dict=model_dict, hyper_dict=hyper_dict,
                       time_dict=time_dict, general_dict=general_dict)
    else:
        run_final_experiment(data_dict=data_dict, model_dict=model_dict, hyper_dict=hyper_dict,
                             time_dict=time_dict, general_dict=general_dict)


if __name__ == "__main__":
    main()
