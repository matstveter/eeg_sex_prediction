#!/usr/bin/env bash
python3 time_series_prediction.py --conf model_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_inc.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_inc2.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_inc3.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_inc4.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf dense_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
