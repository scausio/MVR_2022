import xarray as xr
from gsw.conversions import z_from_p
from distutils.version import LooseVersion
from .base import DataSource


class ArgoSource(DataSource):
    """Intake driver class to handles Argo files

    Attributes:
    -----------
    name: name of the drive
    version: version string for the driver
    container: container type of data sources created by this object
    partition_access: data sources returned by this driver have multiple partitions
    """
    container = 'xarray'
    name = 'argo'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        """Return all of the data from partition id i

        :param i: partition number
        :return data: data from partition id i
        """
        data = xr.open_dataset(self.files[i]).chunk()

        lat = self.metadata.get('coords', {}).get('latitude', 'LATITUDE')
        lon = self.metadata.get('coords', {}).get('longitude', 'LONGITUDE')
        depth = self.metadata.get('coords', {}).get('depth', 'DEPTH')
        time = self.metadata.get('coords', {}).get('time', 'TIME')

        # Backwards compatibility with xarray 0.14
        if LooseVersion(xr.__version__) >= '0.15.0':
            swap_dims = data.swap_dims
        else:
            swap_dims = data.rename_dims
        data = swap_dims({lat: 'POSITION',
                          lon: 'POSITION',
                          time: 'POSITION'})

        data = data.stack(obs=('POSITION', depth)).reset_index('obs').drop(['POSITION', depth])

        try:
            data['DC_REFERENCE'] = data['DC_REFERENCE'].astype(int)
        except ValueError:
            pass

        z = z_from_p(data['PRES'], data[lat])
        data.coords[depth] = xr.DataArray(-z, dims=('obs'))

        # Make sure all coordinates are set as coordinates
        data = data.set_coords(self.metadata.get('coords', {}).values())

        data = data.rename({v: k for k, v in self.metadata.get('coords', {}).items()})

        variables = self.metadata.get('variables')
        if variables:
            data = data.get(list(variables.values()))
        if data is not None:
            data = data.rename({v: k for k, v in self.metadata.get('variables', {}).items()})

        return data

    def read(self):
        """Return all the data in memory in one in-memory container.

        :return: data in memory in an xarray container
        """
        self._load_metadata()
        partitions = [self.read_partition(i) for i in range(self.npartitions)]
        return xr.concat([p for p in partitions if p is not None], dim="obs")
