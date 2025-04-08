#!/usr/bin/env bash

# conda environment
eval "$(conda shell.bash hook)"
conda activate S2C

# main program
# sleep 10
# echo 10s
echo nohup start
cd ~/S2C
python get_se_map_foodai_parallel.py

# Remove container
echo Task done, removing container $CCS_ID
twccli rm ccs -fs $CCS_ID
echo Removed!