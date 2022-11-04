#!/bin/sh

export PATH=$HOME/anaconda2/bin:$PATH
export PYTHONPATH=$HOME/prodqual:$PYTHONPATH

cd $HOME/prodqual
# HDF5 file locking doesn't seem to work well on Athena
export HDF5_USE_FILE_LOCKING=FALSE

luigi --local-scheduler --module tasks $*
