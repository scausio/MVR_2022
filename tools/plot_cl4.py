import matplotlib
matplotlib.use('TkAgg')
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np



base='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20200701_20220930.nc'

depth =0
metric=1 #'number of data values' 'mean of product      ' 'mean of reference    ' 'mean squared error   ' 'variance of product  ' 'variance of reference' 'covariance           '
var='stats_temperature' # 'stats_sst' 'stats_salinity' 'stats_temperature'

ds = xr.open_dataset(base)
print (ds)

print(ds.metric_names[0])
obs=ds.isel(depths=depth, metrics=2)
model=ds.isel(depths=depth,metrics=metric)
time=model.time

#np.arange(len(model.time))
if metric==1:
    plt.plot(time.values, obs[var].isel(forecasts=0).isel(areas=0).values, color='k', label='obs', linestyle='dashed',zorder=10)
for i,fc in enumerate(ds.forecasts):
    lab=['an','fc_1','fc_3','fc_5']
    plt.plot(time,model[var].sel(forecasts=fc).isel(areas=0).values, label=lab[i],zorder=10)
    plt.scatter(time.values, model[var].sel(forecasts=fc).isel(areas=0).values,s=1)
plt.legend(loc='best')
plt.xticks(rotation = 30)
ax=plt.gca()
ax1=ax.twinx()
try:
    ax1.bar(time.values, ds[var].isel(surface=depth,metrics=0,forecasts=3).values[:,0],alpha=0.4,color='k',zorder=1)
except:
    ax1.bar(time.values, ds[var].isel(depths=depth, metrics=0, forecasts=3).values[:, 0], alpha=0.4, color='k', zorder=1)
ax1.set_ylabel('No. observations')
plt.show()

