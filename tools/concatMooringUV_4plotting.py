import xarray as xr
from dateutil.relativedelta import relativedelta
import datetime
import os



startdate='202201'
enddate='202209'
base='/data/inputs/metocean/historical/obs/in_situ/monthly/BlackSea/multiparameter/mooring/{date}/BS_TS_MO_{stat}_{date}.nc'
outpat='/work/opa/bsfs-dev/MVR_2022/auxiliary_data/dev'
stats=['EUXRo01','EUXRo02','EUXRo03']


dates = []
current = datetime.date(int(startdate[:4]), int(startdate[4:]), 1)
end = datetime.date(int(enddate[:4]), int(enddate[4:]), 1)
while current <= end:
    dates.append(datetime.datetime.strftime(current, '%Y%m'))
    current += relativedelta(months=1)

for stat in stats:
    print (stat)
    buffer=[]
    for date in dates:
        print (date)
        ds=xr.open_dataset(base.format(date=date,stat=stat)).isel(DEPTH=2)[['HCDT','HCSP']]

        buffer.append(ds)
        print (buffer[-1])
    xr.concat(buffer,dim='TIME').to_netcdf(os.path.join(outpat,f'{stat}_{startdate}_{enddate}.nc'))


