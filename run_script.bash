#!/usr/bin/env bash
python3 time_series_prediction.py --conf final_run_inception_MC.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_EEGNet_MC.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_weights1.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf final_run_inception_weights2.ini --dict /home/tvetern/datasets/numpy/raw_no_change/

# RANDOM EXPERIMENT
python3 time_series_prediction.py --conf final_run_EEGNet_drop.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
