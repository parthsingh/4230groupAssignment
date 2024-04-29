#!/bin/bash
#SBATCH --job-name=project1
#SBATCH --output=project1.out
#SBATCH --error=project1.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G

# Load necessary modules (if needed)
#module load python

# Activate your virtual environment (if needed)
# source /path/to/your/venv/bin/activate

# Run your Python script
mpiexec -n 1 python final.py