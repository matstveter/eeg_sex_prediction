#!/usr/bin/env bash
python3 time_series_prediction.py --conf kernel_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf model_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf k_fold.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
