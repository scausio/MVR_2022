import xarray as xr
from gsw.conversions import z_from_p
from distutils.version import LooseVersion
from .base import DataSource
import numpy as np

class MooringSource(DataSource):
    container = 'xarray'
    name = 'moor'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        data = xr.open_dataset(self.files[i]).chunk()
        print (self.files[i])

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
        #data["DEPs"] = ('obs', np.zeros(len(data["TIME"].values))+data['DEPH'].values)
        data = data.rename({"DEPH": "DEPTH","DEPH_QC": "DEPTH_QC"})
        data['DC_REFERENCE'] = ('obs', [data.attrs["platform_code"] for i in data["TIME"].values])

        try:
            data = data.drop_dims(['POSITION'])
        except:
            pass
        try:
            data = data.drop('POSITION_QC')
        except:
            pass

        # Make sure all coordinates are set as coordinates
        data = data.set_coords(self.metadata.get('coords', {}).values())
        data = data.rename({v: k for k, v in self.metadata.get('coords', {}).items()})

        variables = self.metadata.get('variables') #['temperature', 'salinity','ssh','u','v'] #
        print (variables)
        buffer=[]
        for i,var in enumerate(list(variables.values())):
            da = data.get(var)
            if isinstance(da, type(None)):
                if buffer:
                    base = buffer[-1].copy()
                    base.values*=np.nan
                    base=base.rename(var)
                    buffer.append(base)
            else:

                buffer.append(da)

        data=xr.merge(buffer)

        if data is not None:
            data = data.rename({v: k for k, v in self.metadata.get('variables', {}).items()})
        print (data)
        depths=np.unique(data.depth.values[data.depth.values>=0])
        buff=[]
        for depth in depths:
            buff.append(data.sel(obs=data.depth==depth).isel(obs=0))
        data = xr.concat(buff,dim='obs')
        return data

    def read(self):
        self._load_metadata()

        partitions=[]
        for i in range(self.npartitions):
            try:
                part = self.read_partition(i)
                partitions.append(part)
            except:
                #print("not valid file")
                continue

        #partitions = [self.read_partition(i) for i in range(self.npartitions)]
        return xr.concat([p for p in partitions if p is not None], dim="obs")
