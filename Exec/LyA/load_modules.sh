#!/bin/bash
module purge
module load python/3.5.1
module load cmake/3.15.3
module load pgi/19.7
module load cuda/10.1
module load openmpi/3.1.4-pgi_19.7
export CUDA_HOME=/projects/opt/centos7/cuda/10.1/
export BLOSC_DIR=/projects/exasky/Nyx_GPU/Exec/LyA/c-blosc/install/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BLOSC_DIR/lib