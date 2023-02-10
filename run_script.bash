#!/usr/bin/env bash

# Single Models
python3 time_series_prediction.py --conf final_run_EEGNet.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_EEGNet_deeper.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_deeper.ini --dict /home/tvetern/datasets/numpy/raw_no_change/

# Ensembles
python3 time_series_prediction.py --conf final_run_EEGNet_MC.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_MC.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_EEGNet_weight_ensemble.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_weight_ensemble.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_depth_ensemble.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
