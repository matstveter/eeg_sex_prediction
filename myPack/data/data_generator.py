import numpy as np
from tensorflow import keras
from myPack.data.data_handling import load_time_series_dataset, shuffle_dataset


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
            data_list, label_list = shuffle_dataset(data_list, label_list)

        return np.array(data_list), np.array(label_list)


def get_all_data_generators(train_id: list, val_id: list, test_id: list, data_dict: dict, time_dict: dict,
                            batch_size, use_conv2d=False):
    """
    1. Creates three DataGenerators: train, val, test
    2. Extracts the model shape based on the input data
    3. Creates a test_dict containing labels and data for all subjects in the test seet

    Args:
        train_id: list over subjects in training set
        val_id: list over subjects in validation set
        test_id: list over subjects in test set
        data_dict: dictionary containing paths and labels
        time_dict: dictionary containing eeg time series specific information (srate, num_windows ...)
        batch_size: Batch size
        use_conv2d: use convolution2D in the models or not

    Returns:
        training_generator: DataGenerator
        validation_generator: DataGenerator
        test_generator: DataGenerator
        test_dict: Dict
        model_shape: shape of input
    """
    training_generator = DataGenerator(list_IDs=train_id,
                                       data_dict=data_dict,
                                       time_slice=time_dict['num_datapoints_per_window'],
                                       start=time_dict['start'],
                                       num_windows=time_dict['num_windows'],
                                       train_set=time_dict['train_set_every_other'],
                                       batch_size=batch_size,
                                       conv2d=use_conv2d)

    validation_generator = DataGenerator(list_IDs=val_id,
                                         data_dict=data_dict,
                                         time_slice=time_dict['num_datapoints_per_window'],
                                         start=time_dict['start'],
                                         num_windows=time_dict['num_windows'],
                                         train_set=False,
                                         batch_size=batch_size,
                                         conv2d=use_conv2d)

    num_test_windows = 40

    test_generator = DataGenerator(list_IDs=test_id,
                                   data_dict=data_dict,
                                   time_slice=time_dict['num_datapoints_per_window'],
                                   start=time_dict['start'],
                                   num_windows=num_test_windows,
                                   train_set=False,
                                   batch_size=batch_size,
                                   conv2d=use_conv2d)

    test_x, _, _ = load_time_series_dataset(participant_list=test_id[0:1],
                                            data_dict=data_dict,
                                            datapoints_per_window=time_dict['num_datapoints_per_window'],
                                            number_of_windows=time_dict['num_windows'],
                                            starting_point=time_dict['start_point'],
                                            conv2d=use_conv2d,
                                            train_set=False)
    model_shape = test_x.shape[1:]

    _, _, test_dict = load_time_series_dataset(participant_list=test_id,
                                               data_dict=data_dict,
                                               datapoints_per_window=time_dict[
                                                   'num_datapoints_per_window'],
                                               number_of_windows=40,
                                               starting_point=time_dict['start_point'],
                                               conv2d=False,
                                               train_set=False,
                                               only_dict=True)
    print(f"Training Generator Length: {len(training_generator)}"
          f"\nValidation Generator Length: {len(validation_generator)}"
          f"\nTest Generator Length: {len(test_generator)}"
          f"\nInput Data Shape: {model_shape}"
          f"\nTest Dict Length: {len(list(test_dict))}")

    return training_generator, validation_generator, test_generator, test_dict, model_shape
