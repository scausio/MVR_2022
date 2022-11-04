import xarray as xr
from distutils.version import LooseVersion
from .base import DataSource

class SLASource(DataSource):
    container = 'xarray'
    name = 'sla'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        data = xr.open_dataset(self.files[i]).chunk()

        lat = self.metadata.get('coords', {}).get('latitude', 'latitude')
        lon = self.metadata.get('coords', {}).get('longitude', 'longitude')
        time = self.metadata.get('coords', {}).get('time', 'time')

        # Backwards compatibility with xarray 0.14
        if LooseVersion(xr.__version__) >= '0.15.0':
            swap_dims = data.swap_dims
        else:
            swap_dims = data.rename_dims
        data = swap_dims({time: 'obs'})

        # Make sure all coordinates are set as coordinates
        data = data.set_coords(self.metadata.get('coords', {}).values())

        data = data.rename({v: k for k, v in self.metadata.get('coords', {}).items()})

        variables = self.metadata.get('variables')
        if variables:
            data = data.get(list(variables.values()))
            data = data.rename({v: k for k, v in self.metadata.get('variables', {}).items()})

        return data

    def read(self):
        self._load_metadata()
        return xr.concat([self.read_partition(i) for i in range(self.npartitions)], dim='obs')
