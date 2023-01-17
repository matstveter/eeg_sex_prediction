import mne
import mne_bids
from mne.externals.pymatreader import read_mat
from myPack.data.read_data import check_dictionary_key
from matplotlib import pyplot as plt
import numpy as np


def interpolate(raw, num_channels):
    raw.plot(n_channels=64)
    plt.show()
    ch_names = raw.info['ch_names']

    assert len(ch_names) < num_channels, "The number of channels in the existing raw file is greater than the " \
                                         "interpolation goal"

    np_data = raw.get_data()
    zero_channel = np.zeros((1, np_data.shape[1]))
    goal_eeg = np.zeros((num_channels, np_data.shape[1]))

    zero_channel_pos = list()
    zero_channel_names = list()

    for i in range((num_channels-len(ch_names))*2):
        if i % 2 != 0:
            zero_channel_pos.append(i)
            zero_channel_names.append(str(i))

    new_ch_names = list()
    k = 0

    for i in range(num_channels):
        if i in zero_channel_pos:
            goal_eeg[i] = zero_channel
            new_ch_names.append(str(k))
        else:
            goal_eeg[k] = np_data[k]
            new_ch_names.append(ch_names[k])
            k += 1

    info = mne.create_info(ch_names=new_ch_names, ch_types='eeg', sfreq=raw.info['sfreq'])
    raw = mne.io.RawArray(goal_eeg, info)
    raw.plot(n_channels=64)
    plt.show()
    raw.info['bads'] = zero_channel_names
    raw.interpolate_bads()
    raw.plot(n_channels=64)
    plt.show()

    return raw, goal_eeg


def create_mne_from_numpy(numpy_array, srate=500, ch_list=None, ch_types="eeg"):
    """
    Function that creates a raw mne_own object from a numpy array
    Args:
        numpy_array: numpy array of shape [channels, time]
        srate: if not 500hz, spesify
        ch_list: if channel names is known, plese supply or else it will be from [0, num_chans] as strings
        ch_types: default as eeg

    Returns:
    rawArray object
    """

    if ch_list == None:
        ch_list = np.arange(0, numpy_array.shape[0])
        ch_list = [str(i) for i in ch_list]

    info = mne.create_info(ch_names=ch_list, sfreq=srate, ch_types=ch_types)

    raw = mne.io.RawArray(numpy_array, info)
    return raw


def get_raw(file_path: str):
    """
    Receives file path and returns a RAW object
    """

    def create_RawArray(dictionary, ch_names):
        srate = check_dictionary_key(dictionary, "srate")
        data = check_dictionary_key(key_dict, "data")
        info_obj = mne.create_info(ch_names, ch_types='eeg', sfreq=srate, verbose=False)

        if len(data) is not len(ch_names):
            return None
        return mne.io.RawArray(data, info_obj, verbose=False)

    file_format = file_path[-4:]

    if file_format == ".edf":
        raw = mne.io.read_raw_edf(input_fname=file_path, verbose=False)
        num_electrodes = raw.info['nchan']
        montage = get_electrode_montage(num_electrodes)
    elif file_format == ".cnt":
        raw = mne.io.read_raw_cnt(input_fname=file_path, verbose=False)
        num_electrodes = raw.info['nchan']
        montage = get_electrode_montage(num_electrodes)
    elif file_format == ".fif":
        raw = mne.io.read_raw_fif(fname=file_path, verbose=False)
        num_electrodes = raw.info['nchan']
        montage = get_electrode_montage(num_electrodes)
    elif file_format == ".set":
        raw = mne.io.read_raw_eeglab(input_fname=file_path, verbose=False, preload=True)
        num_electrodes = raw.info['nchan']
        # montage = get_electrode_montage(num_electrodes)
        return raw
    elif file_format == 'vhdr':
        raw = mne.io.read_raw_brainvision(vhdr_fname=file_path, verbose=False)
        return raw
    elif file_format == ".mat":
        matfile_dictionary = read_mat(file_path)

        if "Raw" in file_path:
            key_dict = check_dictionary_key(matfile_dictionary, "EEG")

            if key_dict == None:
                print("[ERROR] Subject missing EEG data in their file, returning None")
                return None

            nbchan = check_dictionary_key(key_dict, "nbchan")

            montage = mne.channels.read_custom_montage("/media/hdd/age_new/channel_locations.sfp")
            ch_names = list(montage.get_positions()['ch_pos'])

            raw = create_RawArray(key_dict, ch_names)
            if raw is None:
                print(f"[ERROR] Difference in length between data and ch_names, returning None")
                return None

        elif "Preprocessed" in file_path:
            key_dict = check_dictionary_key(matfile_dictionary, "result")
            nbchan = check_dictionary_key(key_dict, "nbchan")
            channel_info = check_dictionary_key(check_dictionary_key(key_dict, 'chaninfo'), 'filecontent')
            eeg = check_dictionary_key(check_dictionary_key(key_dict, 'chanlocs'), 'labels')

            eeg_channel_info = dict()
            non_eeg_channel_info = dict()
            bad_channels = dict()

            for elem in channel_info:
                chans = elem.split("\t")
                if chans[0] in eeg:
                    eeg_channel_info[chans[0]] = list(map(float, chans[1:]))
                elif chans[0] == 'FidNz' or chans[0] == 'FidT9' or chans[0] == 'FidT10':
                    non_eeg_channel_info[chans[0]] = list(map(float, chans[1:]))
                else:
                    bad_channels[chans[0]] = chans[1:]

            raw = create_RawArray(key_dict, eeg)
            montage = mne.channels.make_dig_montage(ch_pos=eeg_channel_info, nasion=non_eeg_channel_info['FidNz'],
                                                    rpa=non_eeg_channel_info['FidT9'],
                                                    lpa=non_eeg_channel_info['FidT10'])

    else:
        raise NotImplementedError("Unrecognized file format: {}".format(file_format))

    raw.set_montage(montage)
    return raw


def create_epochs_from_raw(raw, epochs_duration: int):
    """
    Function to create a Epoch object from a raw object
    """
    return mne.make_fixed_length_epochs(raw, duration=epochs_duration, preload=True, verbose=False)


def get_electrode_montage(num_electrodes: int):
    if num_electrodes == 128:
        montage = mne.channels.make_standard_montage("biosemi128")
    elif num_electrodes == 64:
        montage = mne.channels.make_standard_montage("biosemi64")
    elif num_electrodes == 129:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    else:
        raise NotImplementedError("Number of electrodes does not match with the implemented montages")

    return montage
