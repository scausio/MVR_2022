import yaml
from munch import munchify
import glob
from cftime import num2date, date2num
import datetime, calendar
from dateutil.rrule import rrule, DAILY
import numpy as np
import xarray as xr
from pqtool.utils import *
from argparse import ArgumentParser
import os
from tasks.utils import getConfigurationByID,getMonthsLastDay
from natsort import natsorted

def concatResults():

    files = natsorted(glob.glob(os.path.join(cfg_base.output, 'result', cfg.outname_tmpl.format(product=cfg_base.product,
        start_date='*',
        end_date='*'))))
    start = os.path.basename(files[0]).replace('.nc', '').split('_')[-2]
    end = os.path.basename(files[-1]).replace('.nc', '').split('_')[-1]
    buffer = []
    for f in files:
        buffer.append(xr.open_dataset(f))

    out=xr.concat(buffer, dim='time',data_vars='minimal')
    encoding = {'area_names': {'dtype': 'bytes'},'metric_names': {'dtype': 'bytes'}}

    out.to_netcdf(os.path.join(cfg_base.output, 'class4', cfg.outname_tmpl.format(product=cfg_base.product,
        start_date=start,end_date=end)),encoding=encoding)


def post_class4():

    date_set = list(rrule(freq=DAILY, dtstart=start_date, until=end_date))
    nofile = False
    outFile = os.path.join(cfg_base.output, 'result', cfg.outname_tmpl.format(product=cfg_base.product,
        start_date=datetime.datetime.strftime(start_date, "%Y%m%d"),
        end_date=datetime.datetime.strftime(end_date, "%Y%m%d")))
    if not os.path.exists(outFile):
        outds = xr.Dataset()

        for i, var_name in enumerate(cfg.variables):
            if var_name.startswith('stats_'):
                if 'depths' in cfg.variables[var_name].coords:
                    depth = cfg.dimensions['depths']
                else:
                    depth = cfg.dimensions['surface']

                da = xr.DataArray(data=np.zeros(
                    (len(date_set), cfg.dimensions['forecasts'], depth, cfg.dimensions['metrics'],
                        cfg.dimensions['areas'])).astype(np.float64),
                                  dims=cfg.variables[var_name].coords,
                                  attrs=cfg.variables[var_name].varAttributes,
                                  name=var_name)
                print (da)
                da[:] = np.NaN
                print(var_name)

                for idx, fc in enumerate(cfg.variables['forecasts'].value):
                    files = cfg.variables[var_name].infile.format(output=cfg_base.output,
                                                                  mod_type=list(cfg.forecasts)[idx],
                                                                  ref_date=ref_date)

                    files = expand_wildcards(files)
                    print(files)
                    if len(sorted(glob.glob(files))) > 0:
                        print(files)
                        obs_type=os.path.basename(files).split('_')[2]
                        inds = xr.open_dataset(files)

                        inds = inds.sel(time=slice(start_date, end_date))
                        times = inds['time']

                        count = inds['count' + var_name[5:]]

                        count = count.where(count > 0)
                        mean_model = inds['mean_model' + var_name[5:]].where(count > 0)
                        mean_obs = inds['mean_obs' + var_name[5:]].where(count > 0 & ~mean_model.isnull())
                        mse = inds['mse' + var_name[5:]].where(count > 0)
                        var_model = inds['var_model' + var_name[5:]].where(count > 0)
                        var_obs = inds['var_obs' + var_name[5:]].where(count > 0 & ~var_model.isnull())
                        cov = inds['cov' + var_name[5:]].where(count > 0)

                        metrics = [count, mean_model, mean_obs, mse, var_model, var_obs, cov]

                        dates = times.data.astype('datetime64[D]')
                        try:
                            date_set = [x.date() for x in date_set]
                        except:
                            pass

                        missing = [date for date in date_set if date not in dates]

                        print(len(missing))

                    else:
                        nofile = True
                        missing = date_set

                        print("nofile")

                    for id, var in enumerate(metrics):

                        var = var.squeeze()
                        #     # metric = metric + var_name[5:]
                        if var.dims[0] != 'date':
                            var = var.transpose()

                        if len(missing) > 0:
                            shape = (1,) + var.shape[1:]
                            empty = np.empty(shape)
                            empty[:] = np.NaN
                            for indx in range(len(missing)):
                                # print(date_set.index(missing[indx]))
                                if not nofile:
                                    var = np.insert(var, date_set.index(missing[indx]), empty, axis=0)
                                else:
                                    var = np.empty((len(date_set),) + var.shape[1:])
                                    var[:] = np.NaN
                                    var[date_set.index(missing[indx])] = empty

                        if var_name in ["stats_sst", "stats_sla"]:
                            print (da.shape,var.shape)
                            try:
                                da[:, idx, 0, id, 0] = np.around(var[:,0], decimals=4)
                            except:
                                da[:, idx, 0, id, 0] = np.around(var[:], decimals=4)
                        else:
                            da[:, idx, :, id, 0] = np.around(var, decimals=4)

                    nofile = False
                # remove metric values for sla
                if var_name == "stats_sla":
                    da.values*= np.nan

                outds[var_name] = da.astype(np.float64)

                # getting index of time, depth and area in which not all the forecasts are available, and set them to nan
                msk=np.repeat(np.isnan(np.sum(da.values,axis=1))[:, np.newaxis,:,:,:], da.values.shape[1], axis=1)
                outds[var_name].values[msk] = np.nan


            else:
                if var_name == 'area_names':
                    d = np.array(cfg.variables[var_name].value)
                    #d = d.reshape(1,1)
                elif var_name == 'metric_names':
                    d = np.array(cfg.variables[var_name].value)
                    #d = d.reshape(7, 1)
                elif var_name == 'time':
                    date_set = list(
                        rrule(freq=DAILY, dtstart=start_date, until=end_date))
                    d = date_set
                else:
                    d = np.asarray(cfg.variables[var_name].value)

                da = xr.DataArray(data=d,
                                  dims=cfg.variables[var_name].coords,
                                  attrs=cfg.variables[var_name].varAttributes,
                                  name=var_name)
                outds[var_name] = da

        # encoding
        units = 'seconds since 1970-01-01 00:00'
        calendar = 'standard'
        outds.forecasts.attrs['long_name'] = 'forecast lead time'
        outds.forecasts.attrs['units'] = 'hours'
        outds.depths.attrs['long_name'] = 'depths'
        outds.depths.attrs['positive'] = 'down'
        outds.depths.attrs['units'] = 'm'
        outds.depths.attrs['description'] = 'depth of the base of the vertical layer over which statistics are aggregated'
        outds.attrs['contact'] = 'service-phy@model-mfc.eu'
        outds.attrs['institution'] = "Centro Euro-Mediterraneo sui Cambiamenti Climatici - CMCC, Italy "
        outds.attrs['product'] = f'{cfg_base.product}'
        outds.attrs['start_date'] = datetime.datetime.strftime(start_date, "%Y%m%d")
        outds.attrs['end_date'] = datetime.datetime.strftime(end_date, "%Y%m%d")
        outds.to_netcdf(outFile,
                        encoding={'time': {'dtype': 'f4', 'units': units, 'calendar': calendar, '_FillValue': None},
                                  'depths': {'dtype': 'f4', '_FillValue': None},
                                  'forecasts': {'dtype': 'f4', '_FillValue': None},
                                  'stats_temperature': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_salinity': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_sla': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_sst': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'area_names': {'dtype': 'S25','char_dim_name':'string_length'},
                                  'metric_names': {'dtype': 'S25','char_dim_name':'string_length'}}, unlimited_dims='time')
    #concatResults()



def _getVarsPlatfs(files):
    variables=[]
    platfs=[]
    for f in files:
        f_split=os.path.basename(f).split('_')
        variables.append((f_split[-1]).split('.')[0])
        platfs.append(f_split[1])
    return np.unique(variables), np.unique(platfs)

def post_class2():
    print (cfg.outname_tmpl)
    tmpl="class2_{plat}_*_{ref_date}*_{var}.nc"
    print(os.path.join(cfg_base.output, ref_date, tmpl.format(plat='*', ref_date=ref_date, var='*')))
    files = natsorted(glob.glob(os.path.join(cfg_base.output, ref_date,tmpl.format(plat='*',ref_date=ref_date,var='*'))))
    vars,platfs=_getVarsPlatfs(files)
    for var in vars:
        print (var)
        if var=='V':
            continue
        else:
            for plat in cfg.variables[var].platforms:
                # Here add merging of U and V
                fs = natsorted(glob.glob(os.path.join(cfg_base.output, ref_date,tmpl.format(plat=plat,ref_date=ref_date,var=var))))
                print (os.path.join(cfg_base.output, ref_date,tmpl.format(plat=plat,ref_date=ref_date,var=var)))
                if var=='U':
                    fs_v = natsorted(glob.glob(
                        os.path.join(cfg_base.output, ref_date, tmpl.format(plat=plat, ref_date=ref_date, var='V'))))
                    outname = os.path.join(cfg_base.output, 'class2', cfg.outname_tmpl.format(
                        plat_name=plat, ref_date=ref_date, var_name='UV'))
                else:
                    fs_v=[]
                    outname=os.path.join(cfg_base.output, 'class2', cfg.outname_tmpl.format(
                        plat_name=plat,ref_date=ref_date,var_name=var))
                print (outname)

                if fs:
                    # this is a workaround for mfdataset bug
                    buffer=[]
                    for i,f in enumerate(fs):
                        buffer.append(xr.open_dataset(f))
                        if fs_v:
                            buffer.append(xr.open_dataset(fs_v[i]))


                    out=xr.merge(buffer)
                    # this selects exactly the running month
                    month_idxs = out.groupby('time.month').groups
                    out=out.isel(time=month_idxs[int(ref_date[4:6])])
                    #
                    out.to_netcdf(outname)
                    out.close()
                else:
                    print (f'Variable {var} not exists for platform {plat}')



parser = ArgumentParser(description='Process intermediate data')
parser.add_argument('-c', '--config', default='../example/config.yaml', help='configuration file')
parser.add_argument('-s', '--start_date')
parser.add_argument('-e', '--end_date')
parser.add_argument('-vc', '--validation_class', required=True, help='Class_4 or Class_2')

args = parser.parse_args()
start_date = args.start_date
end_date = args.end_date
vc=args.validation_class
ref_date=start_date[:6]

cfg =getConfigurationByID(args.config,vc)
cfg_base=getConfigurationByID(args.config,'base')

start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
#end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
end_date=datetime.datetime.strptime(getMonthsLastDay(start_date,end_date)[0],"%Y-%m-%d")
if vc=='Class_4':
    post_class4()
else:
    post_class2()

