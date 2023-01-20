#!/usr/bin/env bash
python3 time_series_prediction.py --conf model_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf depth_tester.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf depth_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf dense_tester.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf dense_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf model_tester.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
