# Product quality quick start guide

Running PQ v2.0 for comparison to ARGO is a process that is done in 3 stages:
- Processing ARGO data (i.e. performing the quality selection, averaging ARGO with too high vertical resolution, etc.): this results in a common observational dataset (argo_YYYY.nc) that everyone can use;
- Processing model data: this takes the argo_YYYY.nc file and calculates the model values corresponding to the time and location of the observations, it writes a file modelX_YYYY.nc that contains both the observations and the model values for one (or more) model(s);
- Analysis and plotting: this part takes any number of modelX_YYYY.nc files (e.g. your new model, your old model, a model from a colleague, data from CMEMS, etc.) and it creates the RMSE and bias figures.

The main idea is that within a collaborative development, the argo_YYYY.nc and modelX_YYYY.nc files are kept in a common directory that is accessible by everyone. This makes it easy to grab the results of someone else to compare to. As long as everyone in the group starts from the same argo_YYYY.nc file, all the results are directly comparable. These files are small, so that they are easy to work with in a Jupyter Notebook, but still contain all the individual observations so that we can do more in-depth analysis. If needed, all the observations have also the dc_reference field, so that they can be easily traced back to the original ARGO data.



## Setting up the environment

Setting up the code for stage 1 and 2 on zeus can be done with the following commands:


```
module load anaconda/3.8
conda create --name pqtool python
source activate pqtool
conda install intake xarray netCDF4 pandas dask numpy scipy matplotlib
pip install gsw
git clone git@github.com:CMCC-Foundation/bs-mfc-pq.git
cd bs-mfc-pq
git checkout v2-refactor
python setup.py develop
```

This will set up a conda environment with the required packages and prepare the PQ v2.0 software directory. In the directory bin/ there are the scripts for stage 1 (process_argo.py) and stage 2 (interpolate_model.py). Both are executables that have a -h option to print some syntax information. To define where your observations and model data are located and how the files are structured PQ uses catalog files, these are small structured text files that define the paths and variable names in order to be able to adapt to data in different formats.

## Running a simple task

Using this information stage 1 and 2 can be performed with the scripts mentioned above by running:

```
bsub -q s_short -M 3145728 ./process_argo.py -c catalog.yaml -n argo -s YYYY-01 -e YYYY-01 -o argo_YYYY.nc
bsub -q s_short -M 3145728 ./interpolate_model.py -c catalog.yaml -n modelX -i argo_YYYY.nc -o modelX_YYYY.nc
```

The creates the ARGO observations file for YYYY (argo_YYYY.nc) and compares the model X to the observations, writing the output in modelX_YYYY.nc.

From this point the model file (which contains also the observations themselves) can be moved to a local machine for further analysis in a Jupyter Notebook. 

Download Anaconda from https://www.anaconda.com/products/individual and follow the recipe for zeus above to install the PQ v2.0 (except the module load command). Install Jupyter Notebook with:

```
conda install jupyter notebook
```

At this point you can download one of the example notebooks from github and analyse the model file:

- Plotting individual ARGO profiles and one or more model(s): https://gist.github.com/ejcmcc/cb19b84eeef418cc74cd04f1671f4c2a

- Calculating and plotting metrics for one or more model(s): https://gist.github.com/ejcmcc/72856ebb899087cd12cbf3d97a53582f

- or metrics for only a particular geographic region: https://gist.github.com/ejcmcc/56e15df13ced71b974f375da8332a7cd

## Notes

- As of `intake>=0.6.0` you cannot use `{}` anymore as a wildcard pattern in the `urlpath` field of a catalog entry. You have to name all the patterns, i.e. use `{build_date}` or `{date2}`, even for patterns that you will not use for subsetting.

- The gsw package is required but does not correctly install from `setup.py`, install it manually using `pip install gsw`

