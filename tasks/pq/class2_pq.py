import xarray as xr
import calendar
from datetime import datetime
import numpy as np
import os


def getEmptyDs(outds, conf, var):
    for v in conf.variables[var].vars:
        if not v in conf.coordinates:
            if v == 'model_ssh':
                outds[conf.variables[var].vars[v].outname] = (
                conf.variables[var].vars[v].coords, np.zeros((len(outds.time), len(outds.forecasts)))*np.nan)
            else:
                outds[conf.variables[var].vars[v].outname] = (conf.variables[var].vars[v].coords,
                                                              np.zeros((len(outds.time),
                                                                        len(outds.depth),
                                                                        len(outds.forecasts)))*np.nan)
    return outds

def fillDs(outds,stat_depth_group, conf, var,i_dpt):
    for v in conf.variables[var].vars:
        if not v in conf.coordinates:
            outVar = stat_depth_group[v].values.reshape((-1, 1))

            if v == 'model_ssh':
                print(outVar)
                outds[conf.variables[var].vars[v].outname].values[:] = outVar
            else:
                outds[conf.variables[var].vars[v].outname].values[:,i_dpt] = outVar
    return outds

class Class2():
    def __init__(self,intermediate,ref_date,config,config_base):
        self.config=config
        self.config_base = config_base
        self.intermediate=intermediate
        self.ref_date=ref_date
        self.main()
    def _getStations(self):
        return self.intermediate.groupby('dc_reference')

    def main(self):
        print (self.config)
        stats= self._getStations()
        conf=self.config

        for plat_name, stat_interm in stats:
            print (plat_name,stat_interm.model)
            print (stat_interm)
            lon=np.round(stat_interm.longitude.values[0],4)
            lat = np.round(stat_interm.latitude.values[0],4)
            depths=np.unique(stat_interm['depth'].values)
            print ('depths', depths)
            for var in conf.variables:
                print (var)
                if plat_name in conf.variables[var].platforms:
                    print(f"processing {var} at  {plat_name}")
                    stat_group = stat_interm.groupby('depth')
                    outds = xr.Dataset()
                    for v in conf.variables[var].vars:
                        print(v)
                        if v in conf.coordinates:
                            if v == 'forecasts':
                                outVar = [conf.coordinates['forecasts'][stat_interm.model.values[0]]]
                            elif v == 'depth':
                                outVar = depths
                            elif v == 'time':
                                outVar = stat_interm.groupby('depth')[0][v].values
                            outds[v] = ((v), outVar)
                    outds = getEmptyDs(outds, conf, var)
                    for i_dpt,stat_depth_group in enumerate(stat_group):
                        outds=fillDs(outds, stat_depth_group[1], conf, var,i_dpt)

                    for attr, val in conf.variables[var].vars[v].varAttributes.items():
                        if type(conf.variables[var].vars[v].varAttributes[attr]) is float:
                            val = np.array(val, 'f4')
                        try:
                            outds[v].attrs[attr] = val
                        except:
                            outds[conf.variables[var].vars[v].outname].attrs[attr] = val
                    for attr, val in conf.globalAttributes.items():
                        if attr =='product':
                            val=val.format(product=self.config_base.product)
                        elif attr=='longitude_east':
                            val=val.format(lon=lon)
                        elif attr == 'latitude_north':
                            val = val.format(lat=lat)
                        outds.attrs[attr] = val

                    r_time = datetime.strptime(self.ref_date, "%Y%m")
                    outds.attrs['start_date'] = self.ref_date + '01'
                    outds.attrs['end_date'] = self.ref_date + str(calendar.monthrange(r_time.year, r_time.month)[1])
                    outds.attrs['buoy_name'] = plat_name
                    comp_dims = dict(zlib=True, _FillValue=None)
                    encoding_dims = {dim: comp_dims for dim in outds.dims}
                    encoding_time = {
                        'time': {'zlib': True, '_FillValue': None, 'dtype': int, 'units': "seconds since 1970-01-01 00:00:00",
                                 'calendar': "standard"}}
                    encoding = dict(encoding_time, **encoding_dims) #Class_2_moor_tracer_fc_0_202202.nc
                    outds.to_netcdf(os.path.join(self.config_base.output,self.ref_date,f"class2_{plat_name}_{stat_depth_group[1].model.values[0]}_{outds.attrs['start_date']}_{outds.attrs['end_date']}_{var}.nc"),
                                    encoding=encoding_time) # CHECK ENCODING
                else:
                    print (f'{plat_name} skipped' )