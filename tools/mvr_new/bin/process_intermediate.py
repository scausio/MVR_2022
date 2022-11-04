import xarray as xr
import numpy as np
import pandas as pd
from pqtool.utils import *
from argparse import ArgumentParser
import yaml
from munch import munchify
from tasks.utils import getConfigurationByID
from tasks.pq.class2_pq import Class2
import os

# filtering outliers according to z-score
def reject_outliers(data, obs_type='argo',reference=None, sigma=5.):
    variables = ['temperature', 'salinity']
    if obs_type=='sst':
        variables.remove('salinity')
    for variable in variables:
        bias = data['model_%s' % variable].isel(model=0) - data[variable]
        stdev = float(bias.std())
        mean = float(np.nanmean(bias))

        total = int(data[variable].count())
        data[variable] = data[variable].where(np.abs(bias - mean) < sigma * stdev)
        accepted = int(data[variable].count())
        try:
            print(total, '->', accepted, 100. * accepted / total, (total - accepted), 'rejected')
        except:
            pass
    return data
###
parser = ArgumentParser(description='Process intermediate data')
parser.add_argument('-c', '--config', default='../example/config.yaml', help='configuration file')
parser.add_argument('-o', '--obs_type', default='argo')
parser.add_argument('-m', '--model', default='fc0')
parser.add_argument('-d', '--date', help='reference date')
parser.add_argument('-vc', '--validation_class', required=True, help='Class_4 or Class_2')

args = parser.parse_args()
config = args.config
obs_type = args.obs_type
mod_type = args.model
ref_date = args.date

vc=args.validation_class

cfg_base=getConfigurationByID('conf.yaml','base')
cfg =getConfigurationByID('conf.yaml','Class_2' if vc=='cl2' else 'Class_4' )

outpath=cfg_base.output
in_file=os.path.join(outpath,ref_date,f"{args.validation_class}_{obs_type}_{mod_type}_{ref_date}.nc")
print(in_file)
intermediate = xr.open_dataset(in_file)

if obs_type == "sla":
    intermediate=intermediate.isel(obs=np.isfinite(intermediate.model_ssh.values[0]))
    mdt_file = cfg_base.mdt
    mdt = xr.open_dataset(mdt_file).get(['mdt', 'old_mdt', 'bathymetry'])
    del mdt.coords['lon']
    del mdt.coords['lat']

    coords = {coord: intermediate[coord] for coord in ['longitude', 'latitude']}

    mdt = mdt.interp(coords, method='nearest')
    intermediate = xr.merge([intermediate, mdt])
    intermediate = intermediate.where(intermediate['bathymetry'] > 1000.)
    intermediate['model_sla'] = intermediate['model_ssh'] - intermediate['mdt']
    # this apply changes only at the old version of BSFS.
    intermediate = intermediate.groupby('model').map(swap_mdt)

if obs_type.split('_')[0]=="moor":
    Class2(intermediate,ref_date,cfg,cfg_base)

else:

    intermediate.coords['date'] = intermediate['time'].astype('datetime64[D]')
    try:
       intermediate = intermediate.dropna('obs')
    except:
       pass

    if obs_type == "argo":
        bins = np.array(cfg.variables.depths.value)
        print (intermediate)
        #add z-score filter
        intermediate = intermediate.groupby_bins('depth', bins=np.insert(bins,0,0) ).map(reject_outliers,obs_type='argo',sigma=3.)
        #
        result = intermediate.groupby_bins('depth', bins=np.insert(bins, 0, 0)).apply(mvr_metrics)
        print (result)
        #result = intermediate.groupby_bins('depth', bins=np.insert(bins,0,0) ).apply(mvr_metrics)
        #
        result=result.rename_dims({"depth_bins":"depths"})
        result = result.drop('depth_bins')
    elif obs_type == "sla":
        along_track = True

        intermediate = intermediate.groupby('model').map(unbias_along_track if along_track else unbias)
        result = intermediate.groupby('date').apply(mvr_metrics)

    elif obs_type == "sst":
        #add z-score filter
        intermediate = intermediate.groupby('date').map(reject_outliers,obs_type='sst',sigma=3.)
        #
        result = intermediate.groupby('date').apply(mvr_sst_metrics)
    else:
        exit(f'{obs_type} not implemented yet')
    try:
        result =result.rename({"date":"time"})
    except:
        pass

    comp = dict(_FillValue=None)
    encoding = {var: comp for var in result.data_vars}

    out_file=os.path.join(outpath,ref_date, f"mvr_{args.validation_class}_{obs_type}_{mod_type}_{ref_date}.nc")
    result.to_netcdf(out_file, encoding=encoding)

