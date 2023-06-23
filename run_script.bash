#!/usr/bin/env bash
python3 time_series_prediction.py --conf MC_EEGNet2.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf MC_EEGNet.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf weight_EEGNet.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf single_EEGNet.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/theta_4_8/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/delta_05_4/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/theta_4_8/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/delta_05_4/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/theta_4_8/
python3 time_series_prediction.py --conf single_inception_deeper.ini --dict /home/tvetern/datasets/numpy/delta_05_4/
