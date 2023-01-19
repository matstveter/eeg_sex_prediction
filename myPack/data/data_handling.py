import random

import enlighten
import numpy as np

from myPack.utils import print_dataset

meaning_of_life = 42


def create_split_from_dict(data_dict: dict):
    """
    Function that receives a dictionary and creates (depending on the predictions) a balanced train/test/val split

    data_dict = dictionary containing data
    prediction_key = Which key from the dictionary should be predicted
    split = how should the data be split into train, val, test
    resample_data = How should the dataset be balanced, either balance through resampling or 
    through cut-off

    Returns:
        train, test, val : list = Containing subject keys
    """
    prediction_key = "Sex"

    if len(list(data_dict.keys())) > 2000:
        split = [0.7, 0.2, 0.1]
        large_data = True
    else:
        split = [0.6, 0.2, 0.2]
        large_data = False

    male = list()
    female = list()

    for k, v in data_dict.items():
        if v[prediction_key] == 1:
            female.append(k)
        else:
            male.append(k)

    if len(male) != len(female):
        if len(male) > len(female):
            temp = male
            male = male[0:len(female)]
            # extra_data = temp[len(female):]
        else:
            temp = female
            female = female[0:len(male)]
            # extra_data = temp[len(male):]

    num_participants = len(male) + len(female)

    num_train = int((split[0] * num_participants) / 2)
    num_val = int((split[1] * num_participants) / 2)

    random.seed(meaning_of_life)
    random.shuffle(male)
    random.shuffle(female)

    train = male[0:num_train] + female[0:num_train]
    val = male[num_train:num_train + num_val] + female[num_train: num_train + num_val]
    test = male[num_train + num_val:] + female[num_train + num_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    assert len(list(set(train).intersection(set(val)))) == 0, "ID exists in both train and val"
    assert len(list(set(train).intersection(set(test)))) == 0, "ID exists in both train and test"
    assert len(list(set(test).intersection(set(val)))) == 0, "ID exists in both val and test"

    return train, val, test, large_data


def load_time_series_dataset(participant_list, data_dict, datapoints_per_window, number_of_windows=None,
                             starting_point=0, train_set=False, conv2d=True):
    """
    Function that reads a folder of numpy arrays, based on a participant list, and returns two arrays, which is
    data and labels, for example for trainX/trainY, or valX/valY

    participant_list = list of participants belonging to that array
    data_dict = dictionary containing all information about the participants, most important label and path for numpy
    time_slice = The width of the time series window
    fixed_size = Should the number of time series windows be a certain amount to balance the dataset, or not

    Returns:
        data, labels, dict, where the dictionary contains all data needed later for analysis
    """
    new_dict = dict()
    data_list = list()
    label_list = list()

    skipped_participants = list()

    if train_set:
        number_of_windows = number_of_windows * 2

    pbar = enlighten.Counter(total=len(participant_list), desc="Number of participants", unit="Participants")
    # Loop through the participant list
    for p in participant_list:
        try:
            data = np.load(data_dict[p]['numpy_path'])
            data = data['data']
        except FileNotFoundError:
            print(f"Participant: {p} -> Not found")
            continue

        if number_of_windows is not None and starting_point + (datapoints_per_window * number_of_windows) > data.shape[1]:
            print(f"Data Shape: {data.shape}, not included, skipping...")
            skipped_participants.append(p)
            continue

        if number_of_windows is not None:
            data = data[:, starting_point:starting_point + (datapoints_per_window * number_of_windows)]
        else:
            if data.shape[1] % datapoints_per_window != 0:
                data = data[:, starting_point: starting_point + (int(data.shape[1] -
                                                                     int(data.shape[1] % datapoints_per_window)))]

        temp_data = np.array(np.array_split(data, data.shape[1] / datapoints_per_window, axis=1))
        if train_set:
            # Keep only every other second window
            temp_data = temp_data[::2]

        # Should not be done if 1D convolution?
        if conv2d:
            temp_data = np.expand_dims(temp_data, axis=3)

        temp_dict = {"data": temp_data, 'label': data_dict[p]["Sex"], 'Age': data_dict[p]['Age']}

        new_dict[p] = temp_dict

        labels = [data_dict[p]["Sex"]] * temp_data.shape[0]
        label_list = label_list + labels

        if len(data_list) == 0:
            data_list = temp_data
        else:
            data_list = np.concatenate((data_list, temp_data), axis=0)
        pbar.update()
    data_list, label_list = shuffle_dataset(data_list, label_list)
    print(f"[INFO] Participants skipped: {skipped_participants}")
    return np.array(data_list), np.array(label_list), new_dict


def get_all_data(data_dict: dict, general_dict: dict, time_dict: dict, only_test=False, conv2d=True):
    """  Function which divides, and returns datasets(train, val, test)

    Args:
        conv2d: If this is True -> expand dims on th data to be (batch_size, channels, time, 1), if false
                (batch_size, channels, time)
        time_dict: Dictionary containing time series specific info, such as num segments, and num data points per seg
        only_test: Not interested in the train and val set, only the test set
        data_dict: dictionary containing data and labels
        general_dict: dictionary with saving paths, and other important options

    Returns:
        model dict: with some changes variables according to the data
        train_x, train_y: training set
        val_x, val_y: validation set, used for validating the training and stopping the training
        test_x, test_y, test_dict: Test set, unseen until testing the model, test_dict is used later for the
        majority voting scheme on the test set, to be sure that the subjects are not mixed
    """
    # train_id, val_id, test_id = create_split_from_dict(data_dict, general_dict['prediction'], split=[0.7, 0.2, 0.1])
    train_id, val_id, test_id = create_split_from_dict(data_dict=data_dict,
                                                       prediction_key="Sex",
                                                       split=(0.6, 0.2, 0.2))

    if general_dict['testing']:
        train_id = train_id[0:20]
        val_id = val_id[0:5]
        test_id = test_id[0:5]
    elif only_test:
        train_id = train_id[0:1]
        val_id = val_id[0:1]

    train_x, train_y, train_dict = load_time_series_dataset(participant_list=train_id,
                                                            data_dict=data_dict,
                                                            datapoints_per_window=time_dict['num_datapoints_per_window'],
                                                            number_of_windows=time_dict['num_windows'],
                                                            starting_point=time_dict['start_point'],
                                                            train_set=time_dict['train_set_every_other'],
                                                            conv2d=conv2d)
    val_x, val_y, val_dict = load_time_series_dataset(participant_list=val_id,
                                                      data_dict=data_dict,
                                                      datapoints_per_window=time_dict['num_datapoints_per_window'],
                                                      number_of_windows=time_dict['num_windows'],
                                                      starting_point=time_dict['start_point'],
                                                      train_set=False,
                                                      conv2d=conv2d)
    test_x, test_y, test_dict = load_time_series_dataset(participant_list=test_id,
                                                         data_dict=data_dict,
                                                         datapoints_per_window=time_dict['num_datapoints_per_window'],
                                                         number_of_windows=int(time_dict['num_windows'] * 2),
                                                         starting_point=time_dict['start_point'],
                                                         train_set=False,
                                                         conv2d=conv2d)
    print_dataset(train_y, val_y, test_y, len(train_id), len(val_id), len(test_id))
    return train_x, train_y, val_x, val_y, test_x, test_y, test_dict, test_id


def shuffle_dataset(data: list, labels: list) -> (list, list):
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
    >>> shuffle_dataset(data_input, true_labels)
    ([3, 1, 5, 4, 2], ['c', 'a', 'e', 'd', 'b'])
    """

    # Check that the data and labels have the same length
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    shuffled_dataset = list(zip(data, labels))
    random.seed(42)
    random.shuffle(shuffled_dataset)
    data, labels = zip(*shuffled_dataset)

    return data, labels
