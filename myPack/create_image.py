import os.path

import numpy as np
from PIL import Image
import enlighten
import numpy
import io

from .mne_own.mne_functions import *
from .data.read_data import *
from myPack.utils import save_to_pkl, load_pkl


def create_numpy_arrays(root: str, save_name: str, normalize=False, bandpass=False, downsample=False,
                        sub_select_channels=False):
    save_folder = "/home/tvetern/datasets/numpy/"
    if downsample:
        sampling_rate = 125
    else:
        sampling_rate = 500
    seconds_per_subject = 200
    l_freq = 0.125
    h_freq = 40.0

    if sub_select_channels:
        channel_map = {'Fp1': 22,
                       'Fp2': 9,
                       'F7': 33,
                       'F3': 24,
                       'Fz': 11,
                       'F4': 124,
                       'F8': 122,
                       'FC3': 29,
                       'FCz': 6,
                       'FC4': 111,
                       'T3': 45,
                       'C3': 36,
                       'C4': 104,
                       'T4': 108,
                       'CP3': 42,
                       'CPz': 55,
                       'CP4': 93,
                       'T5': 58,
                       'P3': 52,
                       'Pz': 62,
                       'P4': 92,
                       'T6': 96,
                       'O1': 70,
                       'Cz': 'Cz'}


    try:
        os.mkdir(save_folder + save_name)
    except FileExistsError:
        print("Folder name exists, please write a new folder name")
        save_name = input("New folder name?\n")
        os.mkdir(save_folder + save_name)

    save_folder = save_folder + save_name
    data_folder = save_folder + "/data/"
    os.mkdir(data_folder)

    removed_subjects = []
    modified_subjects = []
    if "age_new" in root:
        removed_subjects = ["NDARBA680RFY", "NDARBB118UDB", "NDARBT747UHM", "NDARBU532YXZ", "NDARCT974NAJ",
                            "NDARCV944JA6", "NDARDH670PXH", "NDARDV310UEG", "NDAREU591YYA", "NDAREX336AC1",
                            "NDARFR849NTP", "NDARGB441VVD", "NDARGF192VD1", "NDARJB233RL7", "NDARJP146GT9",
                            "NDARMM878ZR1", "NDARZD099KWW", "NDARZL855WVA", "NDARHB993EV0", "NDARAC349YUC"]

        modified_subjects = {"NDARAC349YUC": 300, "NDARCU865PBV": 65, "NDARDE283PLC": 120, "NDARDE294HNX": 40,
                             "NDARDJ092YKH": 80, "NDARDN489EXJ": 70, "NDARDX561MRY": 280, "NDAREK918EC2": 70,
                             "NDARFD822TY5": 0, "NDARGG539YGN": 45, "NDARHF351AA6": 60, "NDARHU211PYD": 200,
                             "NDARJY141RFE": 70, "NDARJZ526HN3": 90, "NDARKM635UY0": 110, "NDARLL914UW4": 80,
                             "NDARRP163YRC": 90, "NDARTB883GUN": 365, "NDARTK834FT9": 30, "NDARTX659HAF": 90,
                             "NDARUJ779NM0": 90, "NDARVH033CA4": 45, "NDARWD655AAA": 120, "NDARWW003WWW": 110,
                             "NDARXB889WUB": 180, "NDARXC878WB2": 120, "NDARXF203DCD": 140, "NDARXF358XGE": 180,
                             "NDARXY240WJC": 120, "NDARZU822WN3": 110}

        data_dict = loop_multiple_folders(root)
    elif "prep_child" in root:
        data_dict = read_folder(root)
    else:
        if "christoffer" in root:
            data_dict = read_folder(root)
            new_dict = dict()
            for k, v in data_dict.items():
                temp_dict = dict()
                temp_dict['Age'] = v['age']
                temp_dict['Sex'] = v['sex']
                temp_dict['raw_path'] = v['raw_path']

                new_dict[k] = temp_dict

            data_dict = new_dict

    pbar = enlighten.Counter(total=len(data_dict.keys()), desc="Numpy Arrays saved", unit="Participants")
    missing_info = list()
    i = 0
    for k, v in data_dict.items():
        start = 15

        if "age_new" in root:
            if k in removed_subjects:
                print(f"Removing {k}: because of bad data!")
                missing_info.append(k)
                pbar.update()
                continue

            if k in list(modified_subjects.items()):
                start = int(modified_subjects[k])

        if type(v['raw_path']) is list:
            raw = get_raw(v['raw_path'][0])
        else:
            raw = get_raw(v['raw_path'])

        if raw is None:
            print(f"[ERROR] Raw array is None from subject: {k}!")
            missing_info.append(k)
            pbar.update()
            continue

        if bandpass:
            raw.filter(.25, 25.0, method="iir")
            # raw.filter(l_freq=l_freq, h_freq=h_freq)

        if downsample:
            raw.resample(sfreq=sampling_rate)

        temp_dict = data_dict[k]
        temp_dict['numpy_path'] = data_folder + k + ".npz"
        try:
            temp_dict['Sex'] = int(temp_dict['Sex'])
        except ValueError as e:
            print(f"Skipping because of {e}!")
            missing_info.append(k)
            pbar.update()
            continue
        try:
            temp_dict['Age'] = int(temp_dict['Age'])
        except KeyError:
            temp_dict['Age'] = int(temp_dict['age'])

        if normalize:
            data = raw.get_data()
            data = (data - np.mean(data, axis=-1, keepdims=True)) / (np.std(data, axis=-1, keepdims=True) + 1e-8)
        else:
            data = raw.get_data()

        if sub_select_channels:
            data_array = list()
            for ke, va in (channel_map.items()):
                if va == "Cz":
                    ra = 128
                else:
                    ra = va
                data_array.append(data[ra])
            data = np.array(data_array)

        data_points = (sampling_rate * (start + seconds_per_subject)) - (sampling_rate * start)

        if data.shape[1] < data_points:
            print(f"Too few data points of {k} with {data.shape[1]}")
            missing_info.append(k)
            pbar.update()
            continue

        numpy.savez(data_folder + str(k), data=data[:, (sampling_rate * start):
                                                       (sampling_rate * (start + seconds_per_subject))])
        data_dict[k] = temp_dict

        pbar.update()
        i += 1

    if missing_info:
        for k in missing_info:
            print(f"[ERROR] Removing from dictionary, key: {k}")
            del data_dict[k]

    with open(save_folder + "/info.txt", 'w') as f:
        f.write(f"Number of subjects: {i}\n")
        f.write(f"Removed Subjects: {removed_subjects}\n")
        f.write(f"Removed due to missing raw file or other important information: {missing_info}\n")
        f.write(f"Modified Subjects: starting location: {modified_subjects}")
        f.write(f"\n\nSrate:{sampling_rate}\nStart: {start}\nLength(sec):{seconds_per_subject}\n")
        f.write(f"Normalize: {normalize}, Bandpass(0.125-40): {bandpass}, Downsample(125hz): {downsample}")

    save_to_pkl(data_dict, save_folder)


def plot_images(subject_key: str, raw, path: str):
    """
    Transform a raw array into 6 images, 3 and 3 with different scaling, from a EEG. 3 five-second windows
    :param subject_key:
    :param raw:
    :param path:
    :return:
    """

    raw.plot(start=35.0, duration=5.0, show_scrollbars=False, n_channels=111, scalings='auto', verbose=0)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close()

    raw.plot(start=40.0, duration=5.0, show_scrollbars=False, n_channels=111, scalings='auto', verbose=0)
    img_buf2 = io.BytesIO()
    plt.savefig(img_buf2, format="png")
    plt.close()

    raw.plot(start=45.0, duration=5.0, show_scrollbars=False, n_channels=111, scalings='auto', verbose=0)
    img_buf3 = io.BytesIO()
    plt.savefig(img_buf3, format="png")
    plt.close()

    fig, axarr = plt.subplots(1, 3, figsize=(30, 30))
    fig.suptitle(f"Subject ID: {subject_key}", fontsize=16)
    axarr[0].imshow(Image.open(img_buf))
    axarr[0].set_title("35-40sek")
    axarr[0].axis('off')
    axarr[1].imshow(Image.open(img_buf2))
    axarr[1].set_title("40-45sek")
    axarr[1].axis('off')
    axarr[2].imshow(Image.open(img_buf3))
    axarr[2].set_title("45-50sek")
    axarr[2].axis('off')
    plt.tight_layout()
    plt.savefig("/home/tvetern/test/" + subject_key)
    plt.close()

    img_buf.close()
    img_buf2.close()
    img_buf3.close()
