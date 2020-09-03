# ensemble-neural-network-weather-forecasts
This repository contains the code for our publication "Ensemble methods for neural network-based weather forecasts".

download_era5_z500.py downloads the training data.

era5_compute_normalization_weigths.py computes and saves normalization weights necessary for the training.

train_era5_2.5deg_weynetal_batch.py trains the neural networks

the scripts
era5_ensemble_make_and_eval_forecasts_dropout.py
era5_ensemble_make_and_eval_forecasts_netens.py
era5_ensemble_make_and_eval_forecasts.py


implement the different ensemble methods. they use the trained networks make ensemble forecasts with them, and evaluate the forecasts. note that what in the paper is called "multitrain" is called "netens" in the code.

analyze_and_plot_era5.py plots the results.

