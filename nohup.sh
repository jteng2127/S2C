#!/usr/bin/env bash

# conda environment
eval "$(conda shell.bash hook)"
conda activate S2C

# main program
# sleep 10
# echo 10s
echo Start train
cd ~/S2C
python train.py --name s2c_exp2 --model s2c

# Remove container
echo Task done, removing container $CCS_ID
twccli rm ccs -fs $CCS_ID
echo Removed!