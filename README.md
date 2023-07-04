# Exploring the Potential Benefits of Deep Learning Ensembles and Uncertainty Quantification for Electroencephalography based Sex Prediction

In this paper the aim was to predict sex directly from raw EEG time serie. This repository contains the single model used in the article, as well as different types of ensemble. In addition an approach for calculating the uncertainty, and epoch rejection, explained in the paper. 

PS. This code is tailored to the dataset from the Child Mind Insitute, and major changes have to be made to make it applicable for other datasets. 

The work flow from start to finish:
1. Download the Child Mind Institute dataset, using the script dataset_downloader.py
2. Preprocess and clean the data using the script (PS: MATLAB needed): https://github.com/hatlestad-hall/prep-childmind-eeg
3. Use the second script dataset_creator.py. In this script the lengths of the EEG per subject is chosen, as well as filtering. This will create a folder with a dictionary called labels.plk which contains all informaton about all the subjects in the dataset, that fits the criterion set when created the dataset, and the absolute path to where the actual eeg file will be stored. The EEg file will be stored in the same folder as the labels file, but in a sub-folder called data/. So if the dataset is moved afterwards these paths must be updated.
4. pytho3 time_series_prediction.py --conf config_file.ini --dict /path_to_where_the_label_file_is [--test]
