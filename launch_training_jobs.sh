

for mem in {1..50}; do
sbatch <<EOF
#! /bin/bash

#SBATCH -A SNIC2019-3-611
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1

/pfs/nobackup/home/s/sebsc/miniconda3/bin/python train_era5_2.5deg_weynetal_batch.py ${mem}
EOF
done