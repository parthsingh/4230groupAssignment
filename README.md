##  How to run it
# Locally 
```
conda activate <env name>
conda install mpi4py
conda install numpy
conda install matplotlib
mpiexec -n 1 python final.py
```

# HPC
```
conda activate <env name>
conda install mpi4py
conda install numpy
conda install matplotlib
sbatch s_final.sh
```
