from myPack.create_image import create_numpy_arrays
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating Dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Either: raw or prep")
    parser.add_argument('--normalize', default=False, action='store_true', help="Whether to normalize dataset or not")
    parser.add_argument('--bandpass', default=False, action="store_true", help="To bandpass between 0 and 30")
    parser.add_argument('--downsample', default=False, action="store_true", help="Downsample")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name for dataset!")
    parser.add_argument('--channels_24', default=False, action="store_true", help="Name for dataset!")

    arg = parser.parse_args()

    if arg.dataset == "raw":
        dataset_root: str = "/media/hdd/Datasets/prep_child/"
    elif arg.dataset == "prep":
        dataset_root: str = "/media/hdd/Datasets/age_new/Preprocessed/"
    else:
        raise ValueError(f"Did not recognize dataset: {arg.dataset}")
    
    save_name = arg.dataset_name

    create_numpy_arrays(dataset_root, save_name=save_name, normalize=arg.normalize, bandpass=arg.bandpass,
                        downsample=arg.downsample, sub_select_channels=arg.channels_24)

