import xarray as xr
import numpy as np
from .base import DataSource


class CFSource(DataSource):
    container = 'xarray'
    name = 'cf_compliant'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        data = xr.open_dataset(self.files[i]).chunk()

        lat = self.metadata.get('coords', {}).get('latitude', 'lat')
        lon = self.metadata.get('coords', {}).get('longitude', 'lon')

        #nav_lat = data.coords[lat]
        #nav_lon = data.coords[lon]

        #if nav_lat.ndim == 2 and nav_lon.ndim == 2:
        #    mask = (nav_lon == -1) & (nav_lat == -1)
        #    nav_lat = nav_lat.where(~mask)
        #    nav_lon = nav_lon.where(~mask)
#
        #    if (nav_lat.min(dim='x') == nav_lat.max(dim='x')).all() and (nav_lon.min(dim='y') == nav_lon.max(dim='y')).all():
        #        data.coords[lat] = nav_lat.mean(dim='x').fillna(0)
        #        data.coords[lon] = nav_lon.mean(dim='y').fillna(0)
#
        #        data = data.set_index(x=lon, y=lat)
        #        data = data.rename({'x': lon, 'y': lat})

        data = data.rename({v: k for k, v in self.metadata.get('coords', {}).items()})

        if not np.issubdtype(data['time'].dtype, np.datetime64):
            try:
                from xarray.coding.times import cftime_to_nptime
                data['time'] = cftime_to_nptime(data['time'])
            except ValueError:
                pass

        variables = self.metadata.get('variables', None)
        if variables:
            data = data.get([v for v in variables.values() if v in data])
            data = data.rename({v: k for k, v in variables.items() if v in data})

        return data

    def read(self):
        self._load_metadata()
        return xr.combine_by_coords([self.read_partition(i) for i in range(self.npartitions)],combine_attrs='override')
