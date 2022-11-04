#!/bin/bash
# Modules
module purge
module load curl
module load intel19.5
module load intel19.5/magics
module load intel19.5/eccodes
module load intel19.5/cdo
module load intel19.5/nco
module load intel19.5/boost
module load intel19.5/szip
module load impi19.5
module load impi19.5/hdf5
module load impi19.5/netcdf
module load impi19.5/parallel-netcdf
module load anaconda/3.7
source activate pqtool

export PYTHONPATH=.
