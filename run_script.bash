#!/usr/bin/env bash
python3 time_series_prediction.py --conf conf_inc_dep2.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf conf_inc_dep3.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf conf_inc_dep.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf conf_time.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf conf_exp.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf conf_inc.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf conf_exp.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
