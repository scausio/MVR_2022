import xarray as xr
import calendar
from datetime import datetime
import numpy as np
import os


def getMooringCoords(ds, conf):
    x = ds[conf.moorings.coords.longitude].values[0]
    y = ds[conf.moorings.coords.latitude].values[0]
    z = np.unique(ds[conf.moorings.coords.depth].values)
    return x, y, z


class Class2():
    def __init__(self, intermediate, ref_date, config, config_base):
        self.config = config
        self.config_base = config_base
        self.intermediate = intermediate
        self.ref_date = ref_date
        self.main()

    def _getStations(self):
        return self.intermediate.groupby('dc_reference')

    def main(self):
        print(self.config)
        stats = self._getStations()
        conf = self.config

        for forecast in self.config.forecasts:
            ds_mod = xr.open_mfdataset(
                os.path.join(self.config_base.output, 'ts_cl2', self.config.forecasts[forecast].filename).format(
                    date=f"{self.ref_date}*", p_date=f"{self.ref_date}*"))
            for var in conf.variables:
                for plat_name in self.conf[variables].platforms:
                    print(f"processing {var} at  {plat_name}")
                    moor = xr.open_dataset(conf.moorings.path.format(moor=plat_name, date=f"{self.ref_date}*"))
                    x_moor, y_moor, z_moor = getMooringCoords(ds, conf)
                    ds_mod_sub = ds_mod.sel()
                    outds = xr.Dataset()
                    for v in conf.variables[var].vars:
                        if v in conf.coordinates:
                            if v == 'forecasts':
                                outVar = [conf.coordinates['forecasts'][stat_interm.model.values[0]]]
                            elif v == 'depth':
                                outVar = conf.coordinates.depth['values']
                            else:
                                outVar = stat_interm[v].values
                            outds[v] = ((v), outVar)
                        else:
                            outVar = stat_interm[v].values.reshape((-1, 1, 1))

                            outds[conf.variables[var].vars[v].outname] = (conf.variables[var].vars[v].coords, outVar)

                    for attr, val in conf.variables[var].vars[v].varAttributes.items():
                        if type(conf.variables[var].vars[v].varAttributes[attr]) is float:
                            val = np.array(val, 'f4')
                        try:
                            outds[v].attrs[attr] = val
                        except:
                            outds[conf.variables[var].vars[v].outname].attrs[attr] = val
                    for attr, val in conf.globalAttributes.items():
                        if attr == 'product':
                            val = val.format(product=self.config_base.product)
                        outds.attrs[attr] = val

                    r_time = datetime.strptime(self.ref_date, "%Y%m")
                    outds.attrs['start_date'] = self.ref_date + '01'
                    outds.attrs['end_date'] = self.ref_date + str(calendar.monthrange(r_time.year, r_time.month)[1])
                    outds.attrs['buoy_name'] = plat_name
                    comp_dims = dict(zlib=True, _FillValue=None)
                    encoding_dims = {dim: comp_dims for dim in outds.dims}
                    encoding_time = {
                        'time': {'zlib': True, '_FillValue': None, 'dtype': int,
                                 'units': "seconds since 1970-01-01 00:00:00",
                                 'calendar': "standard"}}
                    encoding = dict(encoding_time, **encoding_dims)  # Class_2_moor_tracer_fc_0_202202.nc
                    outds.to_netcdf(os.path.join(self.config_base.output, self.ref_date,
                                                 f"class2_{plat_name}_{stat_interm.model.values[0]}_{outds.attrs['start_date']}_{outds.attrs['end_date']}_{var}.nc"),
                                    encoding=encoding_time)  # CHECK ENCODING
            else:
                print(f'{plat_name} skipped')
