#!/usr/bin/env bash
python3 time_series_prediction.py --conf testing.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf dense_tester.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf model_tester.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/

python3 time_series_prediction.py --conf run_inc.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_deep.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_time.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_time2.ini --dict /home/tvetern/datasets/numpy/raw_no_change/
python3 time_series_prediction.py --conf run_eegS.ini --dict /home/tvetern/datasets/numpy/raw_no_change/

python3 time_series_prediction.py --conf run_inc.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf run_deep.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf run_time.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf run_time2.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/
python3 time_series_prediction.py --conf run_eegS.ini --dict /home/tvetern/datasets/numpy/raw_24_bandpass_0.25_25/

