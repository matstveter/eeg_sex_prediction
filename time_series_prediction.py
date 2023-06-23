from myPack.final_experiment import run_final_experiments
from myPack.logging_info.parsing import setup_run
import os
from keras.utils.generic_utils import CustomMaskWarning

import warnings
warnings.filterwarnings("ignore", category=CustomMaskWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def main():
    # Get data, read config files for hyperparameters
    data_dict, model_dict, hyper_dict, time_dict, general_dict = setup_run()

    run_final_experiments(data_dict=data_dict, model_dict=model_dict, hyper_dict=hyper_dict,
                          time_dict=time_dict, general_dict=general_dict)


if __name__ == "__main__":
    main()
