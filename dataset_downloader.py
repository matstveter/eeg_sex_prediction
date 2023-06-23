"""
Script for downloading EEG restingstate mat files of Child Mind Institute dataset. 

Required changes:
- download_folder: Where to save the resting state mat files

Optional:
- file_name: Which filename to download from the Child Mind Institute dataset
- eeg_type: Either raw or preprocessed

# Worked December 01.12.2022

Authors: Mats Tveter and Thomas Tveitstøl
"""
import boto3
from botocore import UNSIGNED
from botocore.client import Config

file_name = "RestingState.mat"
eeg_type = "raw"
download_path = "" # Needs to be specified

def main() -> None:
    BUCKET = 'fcp-indi'
    PREFIX = "data/Projects/HBN/EEG"

    s3client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Creating buckets needed for downloading the files
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    s3_bucket = s3_resource.Bucket(BUCKET)

    # Paginator is need because the amount of files exceeds the boto3.client possible maxkeys
    paginator = s3client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    # All subject names will be appended to the following list
    subject_names: list[str] = list()

    number_of_downloaded_subjects = 0
    overlapping_subjects = 0

    download_folder = download_path

    # Looping throught the content of the HBN bucket 
    for page in pages:
        for obj in page['Contents']:
            temp_key = obj['Key']

            # Download only RestingState.mat files in raw folder
            if eeg_type in temp_key and file_name in temp_key:

                # Get subject name from the dictionary key
                name = temp_key.split("/")[4]

                # In case of overlapping, print warning, else download
                if name in subject_names:
                    print("Key already exists! Key: ", name)
                    overlapping_subjects += 1
                else:
                    subject_names.append(name)
                    s3_bucket.download_file(temp_key, download_folder + name + ".mat")
                    number_of_downloaded_subjects += 1


    # Print information about the runs
    print(f"Downloaded Subjects: {number_of_downloaded_subjects}")
    print(f"Overlapping Subjects: {number_of_downloaded_subjects-overlapping_subjects}")


if __name__ == "__main__":
    main()