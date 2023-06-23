import os
import pickle

import enlighten
import mne.io
import numpy
import pandas as pd

def read_csv_tsv(path_to_csv):
    if ".tsv" in path_to_csv:
        labels_df = pd.read_csv(path_to_csv, sep='\t')
    elif ".csv" in path_to_csv:
        labels_df = pd.read_csv(path_to_csv, sep=',')
    else:
        raise ValueError("Unrecognized file type")

    column_names = list(labels_df.columns.values)

    data_dict = dict()
    subjects = list()

    csv_content_dict = labels_df.set_index(column_names[0]).T.to_dict('list')

    for key, val in csv_content_dict.items():
        temp_dict = dict()
        for i, v in enumerate(val):
            temp_dict[column_names[i+1]] = v
        data_dict[key] = temp_dict
        subjects.append(key)

    return data_dict, subjects


def main():
    # HYPERPARAMETERS
    path_to_dataset = "/media/hdd/Datasets/prep_child/"
    save_folder = "/home/tvetern/temp"
    sampling_rate = 500
    seconds_per_subject = 150
    start = 15
    bandpass = False
    freqs = [4.0, 8.0]  # If bandpass, l_freq and h_freq

    # First create dataset folder, then create a sub-folder called data
    save_folder = os.path.join(save_folder, "child_mind_data")
    os.mkdir(save_folder)
    data_folder = os.path.join(save_folder, "data")
    os.mkdir(data_folder)

    files_in_folder = os.listdir(path_to_dataset)
    files_in_folder.sort()

    csv_tsv = list(filter(lambda files_in_folder: ".tsv" in files_in_folder or ".csv"
                                                  in files_in_folder, files_in_folder))[0]
    csv_tsv = csv_tsv
    files_in_folder.remove(str(csv_tsv))

    chans = list(filter(lambda files_in_folder: "all_bad_chans.mat" in files_in_folder, files_in_folder))[0]
    files_in_folder.remove(str(chans))

    data_dict, subject_list = read_csv_tsv(os.path.join(path_to_dataset, csv_tsv))

    pbar = enlighten.Counter(total=len(data_dict.keys()), desc="Number of participants saved", unit="Participants")
    missing_information = list()

    num_subjects = 0
    for f in files_in_folder:

        path_to_subject_data = os.path.join(path_to_dataset, f)

        try:
            temp_dict = data_dict[f.split(".")[0]]
        except KeyError:
            print(f"Can not find the participant in the participant.tsv file...:{f}")
            missing_information.append(f)
            pbar.update()
            continue
        raw = mne.io.read_raw_eeglab(input_fname=path_to_subject_data, verbose=False, preload=True)

        if raw is None:
            print(f"Error: Raw array is none for subject {f}")
            missing_information.append(f)
            pbar.update()
            continue

        if bandpass:
            raw.filter(l_freq=freqs[0], h_freq=freqs[1])

        temp_dict['numpy_path'] = os.path.join(data_folder, f)
        temp_dict['Age'] = int(temp_dict['Age'])
        try:
            temp_dict['Sex'] = int(temp_dict['Sex'])
        except ValueError as e:
            print(f"Key error 'Sex'..skipping participant: {f} ")
            missing_information.append(f)
            pbar.update()
            continue

        numpy_data = raw.get_data()

        # Calculate the amount of data points we want to extract
        data_points = (sampling_rate * (start + seconds_per_subject)) - (sampling_rate * start)

        if numpy_data.shape[1] < data_points:
            print(f"Too few data-points for subject {f}")
            missing_information.append(f)
            pbar.update()
            continue
        
        numpy.savez(temp_dict["numpy_path"], data=numpy_data[:, (sampling_rate * start): 
                                                  (sampling_rate * (start + seconds_per_subject))])
        data_dict[f.split(".")[0]] = temp_dict
        num_subjects += 1

    if missing_information:
        for z in missing_information:
            print("Removing subjects with missing data from the data dictionary")
            del data_dict[z.split(".")[0]]


    with open(os.path.join(save_folder, "dataset_info.text"), 'w') as f:
        f.write(f"Number of subjects: {num_subjects}")
        f.write(f"Sampling rate: {sampling_rate}, Start: {start}, Num seconds per subject: {seconds_per_subject}")
        if bandpass:
            f.write(f"Bandpass with: l_freq:{freqs[0]} and h_freq:{freqs[1]}")
        f.write(f"Subjects removed due to missing data, or other information: {missing_information}")

    dict_file = open(os.path.join(save_folder, 'labels.pkl'), 'wb')
    pickle.dump(data_dict, dict_file)
    dict_file.close()
    print("Dictionary File containing paths and labels for all includes subjects stored in folder!")


if __name__ == "__main__":
    main()