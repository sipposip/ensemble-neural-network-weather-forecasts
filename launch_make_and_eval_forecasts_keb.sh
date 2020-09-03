

for svd_leadtime in 1 2 4 8; do
    sbatch era5_ensemble_make_and_eval_forecasts.py ${svd_leadtime}
done