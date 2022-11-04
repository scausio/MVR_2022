import xarray as xr
import numpy as np
from .base import DataSource


class ShyfemSource(DataSource):
    container = 'xarray'
    name = 'shyfem'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        data = xr.open_dataset(self.files[i]).chunk()

        #coords = self.metadata.get('coords', None)
        #if coords:
        #    data = data.get([v for v in coords.values() if v in data])
        #    data = data.rename({v: k for k, v in coords.items() if v in data})

        #data = data.set_coords(coords)

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

        coords = self.metadata.get('coords', None)

        if coords:
            data = data.set_coords(coords)

        return data

    def read(self):
        self._load_metadata()
        return xr.combine_by_coords([self.read_partition(i) for i in range(self.npartitions)])
