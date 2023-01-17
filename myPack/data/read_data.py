import os
import pickle
import pandas as pd

filename_endings = (".mat", ".edf", ".fif", ".set", ".cnt", ".vhdr")


def locate_files_for_loading_eeg(dict, subjects, path):
    files = dict.fromkeys(subjects)
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if str(f).endswith(filename_endings):
                for sub in subjects:
                    if sub in (dirpath + str(f)):
                        temp1 = files[sub]
                        if temp1 is None:
                            files[sub] = [dirpath + "/" + str(f)]
                        else:
                            files[sub].append(dirpath + str(f))
    empty_keys = list()
    for key2, value in dict.items():
        if files[key2] is None:
            empty_keys.append(key2)
        else:
            value['raw_path'] = files[key2]
            dict[key2] = value
    return dict


def read_folder(root_path, csv_file_path=None):
    """
    Loops through folder and creates a dictionary of all the variables, and also adds the 
    absolute path to the subjects eeg file

    root_path = path to folder
    csv_file_path = Set if the csv file is not present in the folder

    returns: dictionary
    """

    def locate_eeg_files(root_path2, subjects2, dictionary, name=None):
        """ Locates eeg files in root path, adds the complete path to the files, and appends
        this to the dictionary
        """
        files = dict.fromkeys(subjects2)
        for dirpath, dirnames, filenames in os.walk(root_path2):
            for f in filenames:
                if str(f).endswith(filename_endings):
                    for sub in subjects2:
                        if sub in (dirpath + str(f)):
                            temp1 = files[sub]
                            if temp1 is None:
                                files[sub] = [dirpath + str(f)]
                            else:
                                files[sub].append(dirpath + str(f))
        empty_keys = list()
        for key2, value in dictionary.items():
            if files[key2] is None:
                empty_keys.append(key2)
            else:
                value['raw_path'] = files[key2]
                dictionary[key2] = value

        # If the paths of the keys are empty, it means that the files is not present, deleting
        if empty_keys:
            print(f"Missing files for these subjects: {empty_keys}")

        for empty in empty_keys:
            del dictionary[empty]

        return dictionary

    folder_content = os.listdir(root_path)
    folder_content.sort()

    # If there are no content in the folder return None
    if not folder_content:
        return None

    if csv_file_path is None:
        csv_file = list(filter(lambda folder_content: ".tsv" in folder_content or ".csv" in folder_content,
                               folder_content))

        assert len(csv_file) == 1, "Multiple csv files in folder"

        csv_file = csv_file[0]
        folder_content.remove(str(csv_file))
        full_dict, subjects = read_csv_file(root_path + csv_file, folder_content)

    else:
        full_dict, subjects = read_csv_file(csv_file_path)

    folders_in_folder = list(filter(lambda folder_content: "." not in folder_content, folder_content))

    if len(folders_in_folder) == 0:
        filetype = folder_content[0][-4:]

        for key in full_dict:
            temp = full_dict[key]
            temp['raw_path'] = root_path + str(key) + str(filetype)
            full_dict[key] = temp
    else:
        full_dict = locate_eeg_files(root_path, subjects, full_dict)
    return full_dict


def read_csv_file(csv_file_path, subject=None):
    if ".tsv" in csv_file_path:
        seperator = "\t"
    elif ".csv" in csv_file_path:
        seperator = ","
    else:
        print("Supplied file is not a csv or tsv file")
        raise NotImplementedError

    label_dataframe = pd.read_csv(csv_file_path, sep=seperator)
    header_values = list(label_dataframe.columns.values)

    print("Assuming that the first column of the csv file is the ID, if not change the code: {}".format(header_values))

    full_dict = dict()
    csv_content_dict = label_dataframe.set_index(header_values[0]).T.to_dict('list')
    subjects = list()
    for key, value in csv_content_dict.items():
        if subject is not None:
            if (str(key + ".mat") in subject) or (str(key) + ".set" in subject):
                temp_dict = dict()
                for i in range(len(value)):
                    temp_dict[header_values[i + 1]] = value[i]
                full_dict[key] = temp_dict
                subjects.append(key)
        else:
            temp_dict = dict()
            for i in range(len(value)):
                temp_dict[header_values[i + 1]] = value[i]

            full_dict[key] = temp_dict
            subjects.append(key)

    return full_dict, subjects


def loop_multiple_folders(path, dirname=None):
    folders = os.listdir(path)
    folders.sort()

    subjects = list()
    full_dict = dict()

    for folder in folders:
        temp_dict = read_folder(os.path.join(path, folder + "/"))

        if temp_dict is not None:
            sub = list(temp_dict.keys())

            if not subjects:
                subjects = sub
                # full_dict = temp_dict
            else:
                if any(s in subjects for s in sub):
                    print("Removing random duplicate")
                    print("Two similar keys exists in the dictionary")

                    set1 = set(sub)
                    set2 = set(subjects)
                    diff = set1.intersection(set2)
                    for d in diff:
                        del temp_dict[d]
                        print(f"Deleting {d} from dictionary, because it is overlapping!")

                    subjects = list(set(subjects + sub))
                else:
                    subjects = subjects + sub

            full_dict.update(temp_dict)

    return full_dict


def check_dictionary_key(dictionary, key):
    """
    Check if a key exists in dictionary
    """
    key_list = list(dictionary.keys())
    if key in key_list:
        return dictionary[key]
    else:
        print(f"Key: {key}, does not exist in key list: {key_list}")
        return None

