import xarray as xr
from gsw.conversions import z_from_p
from distutils.version import LooseVersion
from .base import DataSource


class ArgoCORASource(DataSource):
    container = 'xarray'
    name = 'argo'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        data = xr.open_dataset(self.files[i]).chunk()
        lat = self.metadata.get('coords', {}).get('latitude', 'LATITUDE')
        lon = self.metadata.get('coords', {}).get('longitude', 'LONGITUDE')
        depth = self.metadata.get('coords', {}).get('depth', 'DEPTH')
        time = self.metadata.get('coords', {}).get('time', 'TIME')
        #print (list(data.keys()))
         
        # Backwards compatibility with xarray 0.14
        if LooseVersion(xr.__version__) >= '0.15.0':
            swap_dims = data.swap_dims
        else:
            swap_dims = data.rename_dims
        #data = swap_dims({lat: 'POSITION',
        #                  lon: 'POSITION',
        #                  time: 'POSITION'})

        retainedList=['PRES','PRES_QC',lat,lon,time,depth, f"{lat}_QC" , f"{lon}_QC", f"{time}_QC", f"{depth}_QC", "TEMP", f"TEMP_QC", 'PSAL', 'PSAL_QC','DC_REFERENCE']
        for v in list(data.keys()):
            if v not in retainedList:
                data=data.drop(v)      
        data = data.stack(obs=('N_PROF',depth )).reset_index('obs').drop(['N_PROF', depth])
       
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
        self._load_metadata()
        partitions=[]
        #for i in range(self.npartitions):
        #     print (i)
        #     partitions.append(self.read_partition(i))

        partitions = [self.read_partition(i) for i in range(self.npartitions)]
        return xr.concat([p for p in partitions if p is not None], dim="obs")
