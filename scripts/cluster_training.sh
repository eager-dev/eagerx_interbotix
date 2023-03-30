#!/bin/bash

#SBATCH --job-name="box_push_training"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=research-3me-cor

module load 2022r2
module load openmpi
module load python/3.8.12
module load py-pip

source /scratch/${USER}/eagerx_venv/bin/activate

srun python /scratch/jelleluijkx/eagerx_interbotix/eagerx_interbotix/train.py >> /scratch/${USER}/box_push_training.log