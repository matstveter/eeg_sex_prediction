import csv
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from typing import Tuple
import pickle
import random

sns.set_theme()
meaning_of_life = 42 # Seeding the random function


def plot_accuracy(history, fig_path, save_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc="lower left")
    plt.ylim(0.0, 1.0)
    plt.savefig(fig_path + save_name + "_training_plot.png")
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(fig_path + save_name + "_training_loss.png")
    plt.close()

    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('Model F1 Score')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(fig_path + save_name + "_f1_score.png")
    plt.close()

    legend = []
    for k, v in history.history.items():
        if k.startswith("auc"):
            plt.plot(history.history[k])
            legend.append('Train')
        if k.startswith("val_auc"):
            plt.plot(history.history[k])
            legend.append('Validation')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='lower left')
    plt.savefig(fig_path + save_name + "_auc.png")
    plt.close()

    plt.plot(history.history['mcc'])
    plt.plot(history.history['val_mcc'])
    plt.title('Model MCC')
    plt.ylabel('MCC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(fig_path + save_name + "_mcc.png")
    plt.close()


def create_folder(path: str, name: str) -> str:
    """
    Creates a folder with the specified name in the specified path.

    This function takes a path and a folder name as input, and it creates a new folder with the specified name in the
    specified path. If the folder already exists, it will not be overwritten.

    Args:
    - path: the path where the new folder should be created
    - name: the name of the new folder

    Returns:
    - save_path: the full path to the new folder

    Examples:
    >>> create_folder('/tmp/', 'new_folder')
    '/tmp/new_folder/'
    """

    # Create the full path to the new folder
    save_path = os.path.join(path + "/", name + "/")

    # Check if the folder already exists
    if not os.path.exists(save_path):
        # Create the folder if it doesn't exist
        os.mkdir(save_path)

    # Return the full path to the folder
    return save_path


def create_run_folder(path: str) -> Tuple:
    """
    Creates a new folder with a unique name based on the current date and time.

    This function takes a path as input, and it creates a new folder with a unique name based on the current date and
    time. The new folder is created in the specified path. The function also creates subfolders for figures and models.

    Args:
    - path: the path where the new folder should be created

    Returns:
    - run_path: the full path to the new folder
    - fig_path: the full path to the subfolder for figures
    - models_path: the full path to the subfolder for models

    Examples:
    >>> create_run_folder('/tmp/')
    ('/tmp/2022_12_06_14_57_22/', '/tmp/2022_12_06_14_57_22/figures/', '/tmp/2022_12_06_14_57_22/models/')
    """

    # Create the new folder with a unique name based on the current date and time
    name = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    run_path = os.path.join(path, name + "/")
    os.mkdir(run_path)

    # Create subfolders for figures and models
    fig_path = os.path.join(run_path, "figures/")
    os.mkdir(fig_path)
    models_path = os.path.join(run_path, "models/")
    os.mkdir(models_path)

    return run_path, fig_path, models_path


def write_to_file(file_name_path: str, string: str, also_print=False) -> None:
    """ Function which takes a filepath as input, and writes a string in that file, can also print the statement,
    given that the argument also_print is set to True. If the file_path_name is not the path to a actual file, an
    error message is printed, and the string is printed in terminal as well

    Args:
        file_name_path: path to file
        string: message to be written in file and/or in terminal
        also_print: Should the string be printed in the terminal

    Returns:
        None
    """
    if os.path.isfile(file_name_path):
        f = open(file_name_path, "a")
        f.write(string + "\n")
        f.close()

        if also_print:
            print(string)
    else:
        print(f"[ERROR] Path given is not a file {file_name_path}, printing the string and returns!")
        print(string)


def print_dataset(train_y: numpy.ndarray, val_y: numpy.ndarray, test_y: numpy.ndarray, num_train, num_val, num_test):
    """ Printing the class distribution in percentages

    Args:
        num_test:
        num_val:
        num_train:
        train_y: Training set ground truth
        val_y: Validation set ground truth
        test_y: Test set ground truth

    Returns:
        None
    """
    print("***************************************************************************************")
    mal = list(train_y).count(0)
    print(f"[INFO] Percentage of male in train data: {(mal / len(train_y)) * 100:.2f} over {len(train_y)} samples, "
          f"num_subjects = {num_train}")
    mal = list(val_y).count(0)
    print(f"[INFO] Percentage of male in validation data: {(mal / len(val_y)) * 100:.2f} over {len(val_y)} samples, "
          f"num_subjects: {num_val}")
    mal = list(test_y).count(0)
    print(f"[INFO] Percentage of male in testing data: {(mal / len(test_y)) * 100:.2f} over {len(test_y)} samples, "
          f"num_subjects: {num_test}")
    print("***************************************************************************************")


def save_to_pkl(data_dict, path, name=None):
    """
    Saving a dictionary to a pkl file
    """
    if name is None:
        try:
            dict_file = open(path + "/labels.pkl", 'wb')
            pickle.dump(data_dict, dict_file)
            dict_file.close()
        except:
            print("Saving as pickle did not work")
    else:
        try:
            dict_file = open(path + "/" + name + ".pkl", 'wb')
            pickle.dump(data_dict, dict_file)
            dict_file.close()
        except:
            print(f"Saving {name} as pickle did not work")


def load_pkl(path):
    """
    Loading a pkl file with a dictionary to a dictionary
    """
    try:
        dict_file = open(path + "/labels.pkl", 'rb')
        data_dict = pickle.load(dict_file)
        dict_file.close()
        return data_dict
    except:
        print("Could not load label file as pkl")


def shuffle_two_arrays(data: list, labels: list) -> (list, list):
    """
    Shuffles the data and labels in the same order.

    This function takes a dataset and a set of labels as input, and it shuffles the data and labels in the same order.
    This can be useful for creating a randomized training set for machine learning algorithms.

    Args:
    - data: a list or array of data samples
    - labels: a list or array of labels for the data samples

    Returns:
    - data: the input data, shuffled in the same order as the labels
    - labels: the input labels, shuffled in the same order as the data

    Raises:
    - ValueError: if the data and labels have different lengths

    Examples:
    >>> data_input = [1, 2, 3, 4, 5]
    >>> true_labels = ['a', 'b', 'c', 'd', 'e']
    >>> shuffle_two_arrays(data_input, true_labels)
    ([3, 1, 5, 4, 2], ['c', 'a', 'e', 'd', 'b'])
    """

    # Check that the data and labels have the same length
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    shuffled_dataset = list(zip(data, labels))
    random.seed(meaning_of_life)
    random.shuffle(shuffled_dataset)
    data, labels = zip(*shuffled_dataset)

    return data, labels
