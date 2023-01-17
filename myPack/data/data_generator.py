import numpy as np
from tensorflow import keras
from myPack.data.data_handling import shuffle_dataset


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, data_dict, time_slice, train_set, batch_size, num_windows, start, shuffle=True,
                 large_data=False, conv2d=True):

        self.indexes = None
        self._batch_size = batch_size
        self._subject_ids = list_IDs
        self._shuffle = shuffle
        self._num_windows = num_windows
        self._start = start
        self._large_data = large_data

        # For loading data
        self._data_dict = data_dict
        self._time_slice = time_slice
        self._train = train_set
        self.on_epoch_end()
        self._use_conv2d = conv2d

    def __len__(self):
        return int(np.floor(len(self._subject_ids) / self._batch_size))

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

            if self._train and (self._start + (self._time_slice*self._num_windows*data_am)) <= data.shape[1]:
                temp_data = data[:, self._start: (self._start + (self._num_windows*self._time_slice*data_am))]
                temp_data = np.array(np.array_split(temp_data, (temp_data.shape[1] / self._time_slice), axis=1))
                temp_data = temp_data[::data_am]
            else:
                temp_data = data[:, self._start: (self._start + (self._num_windows*self._time_slice))]
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
