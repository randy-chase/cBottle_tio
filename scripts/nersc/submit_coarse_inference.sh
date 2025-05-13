#!/bin/bash

#SBATCH --account=trn006
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --module=gpu,nccl-plugin
#SBATCH --image=registry.nersc.gov/m4935/cbottle
#SBATCH -o cBottle_inference_1node_%j.out 
#SBATCh -e cBottle_inference_1node_%j.err

set -x

# run training job
cd /pscratch/sd/a/akshay13/cBottle/
srun --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 shifter \
    bash -c "
    unset NCCL_CROSS_NIC
    pip install -e .
    python scripts/inference_coarse.py \
        /global/cfs/cdirs/m4331/tge/cBottle-3d.zip \
        inference_output \
        --sample.min_samples 1
    "
