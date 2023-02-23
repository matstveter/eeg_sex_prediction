import numpy as np
from tensorflow import keras
from myPack.utils import shuffle_two_arrays
import random
import enlighten

meaning_of_life = 42  # Used for seeding the random function


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, data_dict, time_slice, train_set, batch_size, num_windows, start, conv2d,
                 shuffle=True):

        self.indexes = None
        self._batch_size = batch_size
        self._subject_ids = list_IDs
        self._shuffle = shuffle
        self._num_windows = num_windows
        self._start = start

        # For loading data
        self._data_dict = data_dict
        self._time_slice = time_slice
        self._train = train_set
        self.on_epoch_end()
        self._use_conv2d = conv2d

    def __len__(self):
        return int(np.ceil(len(self._subject_ids) / self._batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self._batch_size: (index + 1) * self._batch_size]

        # Find list of IDs
        subject_ids_temp = [self._subject_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(subject_ids_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self._subject_ids))
        if self._shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, subject_ids_temp):

        label_list = list()
        data_list = list()
        for p in subject_ids_temp:
            data = np.load(self._data_dict[p]['numpy_path'])
            data = data['data']

            if self._time_slice * self._num_windows > data.shape[1]:
                print(f"Skipping participant due to missing data: {p} {data.shape}")
                continue
            data_am = 2

            if self._train and (self._start + (self._time_slice * self._num_windows * data_am)) <= data.shape[1]:
                temp_data = data[:, self._start: (self._start + (self._num_windows * self._time_slice * data_am))]
                temp_data = np.array(np.array_split(temp_data, (temp_data.shape[1] / self._time_slice), axis=1))
                temp_data = temp_data[::data_am]
            else:
                temp_data = data[:, self._start: (self._start + (self._num_windows * self._time_slice))]
                temp_data = np.array(np.array_split(temp_data, temp_data.shape[1] / self._time_slice, axis=1))

            if self._use_conv2d:
                temp_data = np.expand_dims(temp_data, axis=3)

            labels = [self._data_dict[p]["Sex"]] * temp_data.shape[0]
            label_list = label_list + labels

            if len(data_list) == 0:
                data_list = temp_data
            else:
                data_list = np.concatenate((data_list, temp_data), axis=0)

        if self._batch_size > 1:
            data_list, label_list = shuffle_two_arrays(data_list, label_list)

        return np.array(data_list), np.array(label_list)


def get_all_data_and_generators(*, data_dict: dict, time_dict: dict, batch_size, train_id: list = None,
                                val_id: list = None, test_id: list = None, use_conv2d=False, only_test=False,
                                load_test_dict=True):
    """
    * Splits the data evenly among the classes
    * Creates three DataGenerators: train, val, test
    * Extracts the model shape based on the input data
    * Creates a test_dict containing labels and data for all subjects in the test set

    Args:
        val_id: if None, get it, else it contains id
        test_id: subjects in the test set, if none extract
        train_id: subjects in training set
        only_test: if test mode is enabled, meaning that we only use a small part of the data for quicker testing
        data_dict: dictionary containing paths and labels
        time_dict: dictionary containing eeg time series specific information (srate, num_windows ...)
        batch_size: Batch size
        use_conv2d: use convolution2D in the models or not
        load_test_dict: whether to load test dictionary

    Returns:
        training_generator: DataGenerator
        validation_generator: DataGenerator
        test_generator: DataGenerator
        test_dict: Dict
        model_shape: shape of input
    """
    # ---- FUNCTION PARAMETERS ---- #
    num_test_windows = 40  # Number of windows for the test set
    # ----------------------------- #

    # Creating a split
    # if train_id is None:
    #     print("NOT RUNNING K-FOLD TRAINING SET")
    #     train_id, val_id, test_id = create_split_from_dict(data_dict)
    #     print(f"Train ID: {len(train_id)}, Val ID: {len(val_id)}, Test ID: {len(test_id)}")

    if only_test:
        train_id = train_id[0:2]
        val_id = val_id[0:1]
        test_id = test_id[0:1]


    training_generator = DataGenerator(list_IDs=train_id,
                                       data_dict=data_dict,
                                       time_slice=time_dict['num_datapoints_per_window'],
                                       start=time_dict['start_point'],
                                       num_windows=time_dict['num_windows'],
                                       train_set=time_dict['train_set_every_other'],
                                       batch_size=batch_size,
                                       conv2d=use_conv2d)

    validation_generator = DataGenerator(list_IDs=val_id,
                                         data_dict=data_dict,
                                         time_slice=time_dict['num_datapoints_per_window'],
                                         start=time_dict['start_point'],
                                         num_windows=time_dict['num_windows'],
                                         train_set=False,
                                         batch_size=batch_size,
                                         conv2d=use_conv2d)

    test_generator = DataGenerator(list_IDs=test_id,
                                   data_dict=data_dict,
                                   time_slice=time_dict['num_datapoints_per_window'],
                                   start=time_dict['start_point'],
                                   num_windows=num_test_windows,
                                   train_set=False,
                                   batch_size=batch_size,
                                   conv2d=use_conv2d)

    # This is simply used for getting the model shape of a single data-point, test_id should be [0:1]!!!
    test_x, _, _ = load_time_series_dataset(participant_list=test_id[0:1],
                                            data_dict=data_dict,
                                            datapoints_per_window=time_dict['num_datapoints_per_window'],
                                            number_of_windows=time_dict['num_windows'],
                                            starting_point=time_dict['start_point'],
                                            conv2d=use_conv2d)
    model_shape = test_x.shape[1:]

    if load_test_dict:
        _, _, test_dict = load_time_series_dataset(participant_list=test_id,
                                                   data_dict=data_dict,
                                                   datapoints_per_window=time_dict[
                                                       'num_datapoints_per_window'],
                                                   number_of_windows=num_test_windows,
                                                   starting_point=time_dict['start_point'],
                                                   conv2d=False,
                                                   only_dict=True)
    else:
        test_dict = None

    print(f"Training Generator Length: {len(training_generator)}"
          f"\t Validation Generator Length: {len(validation_generator)}"
          f"\t Test Generator Length: {len(test_generator)}"
          f"\nInput Data Shape: {model_shape}")
    if load_test_dict:
        print(f"Test Dict Length: {len(list(test_dict))}")

    return training_generator, validation_generator, test_generator, test_dict, model_shape


def create_split_from_dict(data_dict: dict, data_split=None, k_fold=False):
    """
    Function that receives a dictionary and creates (depending on the predictions) a balanced train/test/val split

    data_dict = dictionary containing data
    data_split = None if no split is selected, then it will be done automatically according to the number of subjects
    k_fold = if this function should only return the sub id for females and males respectively

    Returns:
        train, test, val : list = Containing subject keys
    """

    if data_split is None:
        if len(list(data_dict.keys())) > 2000:
            split = [0.6, 0.2, 0.2]
        else:
            split = [0.6, 0.2, 0.2]
    else:
        split = data_split

    male = list()
    female = list()

    for k, v in data_dict.items():
        if v["Sex"] == 1:
            female.append(k)
        else:
            male.append(k)

    if len(male) != len(female):
        if len(male) > len(female):
            # temp = male
            male = male[0:len(female)]
            # extra_data = temp[len(female):]
        else:
            # temp = female
            female = female[0:len(male)]
            # extra_data = temp[len(male):]

    if k_fold:
        # random.seed(meaning_of_life)
        random.shuffle(female)
        random.shuffle(male)
        return female, male

    num_participants = len(male) + len(female)
    print(f"Total number of subjects used: {num_participants}")

    num_train = int((split[0] * num_participants) / 2)
    num_val = int((split[1] * num_participants) / 2)

    # random.seed(meaning_of_life)
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

    return train, val, test


def load_time_series_dataset(participant_list, data_dict, datapoints_per_window, number_of_windows=None,
                             starting_point=0, conv2d=True, only_dict=False):
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

    if len(participant_list) > 10:
        pbar = enlighten.Counter(total=len(participant_list), desc="Number of participants", unit="Participants")
    # Loop through the participant list
    for p in participant_list:
        try:
            data = np.load(data_dict[p]['numpy_path'])
            data = data['data']
        except FileNotFoundError:
            print(f"Participant: {p} -> Not found")
            continue

        if number_of_windows is not None and starting_point + (datapoints_per_window * number_of_windows) > data.shape[
            1]:
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

        # Should not be done if 1D convolution?
        if conv2d:
            temp_data = np.expand_dims(temp_data, axis=3)

        temp_dict = {"data": temp_data, 'label': data_dict[p]["Sex"], 'Age': data_dict[p]['Age']}

        new_dict[p] = temp_dict

        if not only_dict:
            labels = [data_dict[p]["Sex"]] * temp_data.shape[0]
            label_list = label_list + labels

            if len(data_list) == 0:
                data_list = temp_data
            else:
                data_list = np.concatenate((data_list, temp_data), axis=0)

        if len(participant_list) > 10:
            pbar.update()

    if not only_dict:
        data_list, label_list = shuffle_two_arrays(data_list, label_list)

    if len(skipped_participants) > 0:
        print(f"[INFO] Participants skipped: {skipped_participants}")
    return np.array(data_list), np.array(label_list), new_dict


def create_k_folds(*, data_dict: dict, num_folds=5):
    def duplicate_check(subs: list) -> None:
        """ Asserts that no subjects exists within several of the folds
        Args:
            subs: list of list of subjects [[sub1, sub2],[sub3,sub4]]
        """
        batch_len = list()
        # Assert that no subject is in multiple lists
        seen_values = set()
        for lst in subs:
            batch_len.append(len(lst))
            for value in lst:
                if value in seen_values:
                    raise AssertionError("Duplicate Subject found {}".format(value))
                seen_values.add(value)
        print(f"K-Fold Sizes: {batch_len} = {sum(batch_len)} -> No duplicates within the folds found!")

    # Get two lists, containing females and males separately
    female, male = create_split_from_dict(data_dict=data_dict, k_fold=True)

    # Find the length of each batch based on the length of either male and female divided by num_folds
    num_per_batch = len(female) // num_folds

    # Pre-create the generators -> train(3 batches), validation(1 batch), test(1 batch)
    subject_batch = []

    for i in range(num_folds):
        start_idx = i * num_per_batch
        end_idx = start_idx + num_per_batch

        temp_list = female[start_idx:end_idx] + male[start_idx:end_idx]
        subject_batch.append(temp_list)

    # Check if there are duplicates within each of the folds
    duplicate_check(subject_batch)

    train_subject_list = []
    val_subject_list = []
    test_subject_list = []

    # Create the generator pairs, create generators and store them in lists
    for n in range(num_folds):
        index_list = [0, 1, 2, 3, 4]
        # Get subjects for the test set
        temp_test = subject_batch[n]
        if n == num_folds - 1:
            val_index = 0
        else:
            val_index = n + 1
        temp_val = subject_batch[val_index]
        index_list.remove(n)
        index_list.remove(val_index)
        temp_train = [element for i in index_list for element in subject_batch[i]]
        duplicate_check(subs=[temp_train, temp_val, temp_test])

        train_subject_list.append(temp_train)
        val_subject_list.append(temp_val)
        test_subject_list.append(temp_test)

    return train_subject_list, val_subject_list, test_subject_list



