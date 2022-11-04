import xarray as xr
from utils import *
import yaml
import calendar
from munch import munchify
from datetime import datetime
from argparse import ArgumentParser
import yaml
import os
from munch import Munch
from utils import getConfigurationByID,getValidPoints,getValidNearest
from glob import glob

parser = ArgumentParser(description='BlackSea Class 2')

parser.add_argument('-r', '--ref_date', help='reference period')
args = parser.parse_args()
ref_date = args.ref_date

config = getConfigurationByID('conf.yaml', 'Class_2')
config_base = getConfigurationByID('conf.yaml', 'base')

for forecast in config.forecasts:
    bs_path=config.forecasts[forecast].out_dir.format(output=config_base.output)
    print (os.path.join(bs_path,config.forecasts[forecast].filename.format(date=f'{ref_date}*',p_date='*')))
    bs_file = glob(os.path.join(bs_path,config.forecasts[forecast].filename.format(date=f'{ref_date}*',p_date='*')))
    model = xr.open_mfdataset(bs_file)
    for var in config.variables:
        stats= config.variables[var].platforms
        for stat in stats:
            outds=xr.Dataset()
            obs_file = config.base_moorings.format(plat_name=stat, ref_date=ref_date)
            obs = xr.open_dataset(obs_file)

            obs_lon = obs.attrs['last_longitude_observation']
            obs_lat = obs.attrs['last_latitude_observation']
            depths=np.unique(obs.DEPH.values)
            depths=depths[depths>0]
            bs_var=config.variables
            try:
                var=model.thetao
            except:
                var=model.votemper
            points=getValidPoints(var.values, model.longitude.values,model.latitude.values)
            new_coords=getValidNearest(points,np.array((obs_lon,obs_lat)).T)

            bs_short = model.sel(lon=new_coords[0], lat=new_coords[1], depth=depths)
            print (bs_short)
            #obs_variable = obs[obs_var].isel(DEPTH=2)
            """
            
            for var in config.variables[var_name].vars:
                if var in config.common_vars:
                    if var in bs_variable.coords:
                        outVar = castToList(bs_variable[var].values)
                    else:
                        outVar = castToList(config.variables[var_name].vars[var].value)
                else:
                    outVar = np.reshape(bs_variable.values, (bs_variable.shape[0], 1, 1))
            
                outds[var] = (config.variables[var_name].vars[var].coords, outVar)
                # Write attributes
                for attr, val in config.variables[var_name].vars[var].varAttributes.items():
                    if type(config.variables[var_name].vars[var].varAttributes[attr]) is not float:
                        outds[var].attrs[attr] = val
                    else:
                        val = np.array(val, 'f4')
                        outds[var].attrs[attr] = val
            
            # Global Attributes
            for attr, val in config.globalAttributes.items():
                outds.attrs[attr] = val
            
            r_time = datetime.strptime(ref_date, "%Y%m")
            outds.attrs['start_date'] = ref_date + '01'
            outds.attrs['end_date'] = ref_date + str(calendar.monthrange(r_time.year, r_time.month)[1])
            outds.attrs['buoy_name'] = plat_name
            comp_dims = dict(zlib=True, _FillValue=None)
            encoding_dims = {dim: comp_dims for dim in outds.dims}
            encoding_time = {'time': {'zlib': True, '_FillValue': None, 'dtype': int, 'units': "seconds since 1970-01-01 00:00:00",
                                      'calendar': "standard"}}
            encoding = dict(encoding_time, **encoding_dims)
            outds.to_netcdf(config.general.out_file.format(plat_name=plat_name, ref_date=ref_date, var_name=var_name),
                            encoding=encoding_time)
                            """