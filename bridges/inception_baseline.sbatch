#!/bin/bash
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p GPU
#SBATCH --gres=gpu:p100:2
#SBATCH -t 8:00:00

# total time for method 2 at 32 al iterations should be about:  19.692307692307693

echo "Running MedAL on hostname: $(hostname)"
date

# load environment
module load AI/anaconda3-5.1.0_gpu.2018-08 
source activate $AI_ENV
cd $SCRATCH/medal_improvements
pwd

cd new
python -m medal


# clear previous model state
# rm Model_save.hdf5
# python Script.py

echo finished training MedAL
date
