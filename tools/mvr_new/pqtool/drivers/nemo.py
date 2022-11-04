import xarray as xr
import pandas as pd
import numpy as np
from .base import DataSource



def _detect_mask(lat, lon):
    '''
    Auto-detect out-of-domain coordinates created by NEMO 4.0.
    
    param lat: latitude values
    param lon: longitude values
    return: False or correct mask
    '''
    shape = lat.shape
    if lon.shape != shape:
        return False, np.nan

    lat = np.asanyarray(lat).ravel() 
    lon = np.asanyarray(lon).ravel() 

    index = pd.MultiIndex.from_arrays([lon, lat])
    mask = index.duplicated(keep=False)

    if mask.any():
        fill = np.unique([lat[mask], lon[mask]])
        if len(fill) == 1:
            return mask.reshape(shape), fill[0]

    return False, np.nan


class NemoSource(DataSource):
    '''
    ## Nemo Driver
        
    Intake driver class to handles NEMO files
        
    Attributes:
    -----------
    name: name of the drive
    version: version string for the driver
    container: container type of data sources created by this object
    partition_access: data sources returned by this driver have multiple partitions
    '''
    container = 'xarray'
    name = 'nemo'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        """Return all of the data from partition id i

        :param i: partition number
        :return data: data from partition id i
        """
        data = xr.open_dataset(self.files[i]).chunk()

        if not any([v in data.data_vars for v in self.metadata.get('variables', {}).values()]):
            return None

        lat = self.metadata.get('coords', {}).get('latitude', 'nav_lat')
        lon = self.metadata.get('coords', {}).get('longitude', 'nav_lon')

        nav_lat = data.coords[lat]
        nav_lon = data.coords[lon]

        if nav_lat.ndim == 2 and nav_lon.ndim == 2:
            mask, fill = _detect_mask(nav_lat, nav_lon)
            nav_lat = nav_lat.where(~mask)
            nav_lon = nav_lon.where(~mask)

            if (nav_lat.min(dim='x') == nav_lat.max(dim='x')).all() and (nav_lon.min(dim='y') == nav_lon.max(dim='y')).all():
                data.coords[lat] = nav_lat.mean(dim='x').fillna(fill)
                data.coords[lon] = nav_lon.mean(dim='y').fillna(fill)

                data = data.set_index(x=lon, y=lat)
                data = data.rename({'x': lon, 'y': lat})

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
        """Return all the data in memory in one in-memory container.

        :return: data in memory in an xarray container
        """
        self._load_metadata()
        partitions = [self.read_partition(i) for i in range(self.npartitions)]
        return xr.combine_by_coords([p for p in partitions if p is not None], combine_attrs='override')
