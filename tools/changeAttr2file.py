import xarray as xr

#ds=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/daySince_product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20200701_20220831.nc')
#print (ds.time)
#units = 'seconds since 1970-01-01 00:00'
#calendar = 'standard'
inname='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20200701_20220831.nc_str1'
#ds.to_netcdf(inname,encoding={'time': {'dtype': 'f4', 'units': units, 'calendar': calendar, '_FillValue': None}})

#
# ibi=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/product_quality_stats_IBI_ANALYSIS_FORECAST_PHYS_005_001_b_20210901_20210930.nc')
# print (ibi)
# exit()

outname='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20200701_20220831_ok.nc'

ds=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class4/product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20220901_20220930.nc')
ds_or=xr.open_dataset(inname)
ds_copy=ds.copy()

ds_copy=ds_copy[['forecasts','depths']]
ds_copy['area_names']=ds_or['area_names'].isel(string_length=0)
ds_copy['metric_names']=ds_or['metric_names'].isel(string_length=0)
ds_copy['stats_salinity']=ds_or['stats_salinity']
ds_copy['stats_sst']=ds_or['stats_sst']
ds_copy['stats_sla']=ds_or['stats_sla']
ds_copy['stats_temperature']=ds_or['stats_temperature']
ds_copy.attrs['start_date'] = '20200701'
ds_copy.attrs['end_date'] = '20220831'
units = 'seconds since 1970-01-01 00:00'
calendar = 'standard'

ds_copy.forecasts.attrs['long_name'] = 'forecast lead time'
#ds_copy.forecasts.attrs['units'] = 'hours'
ds_copy.depths.attrs['long_name'] = 'depths'
ds_copy.depths.attrs['positive'] = 'down'
ds_copy.depths.attrs['units'] = 'm'
ds_copy.depths.attrs['description'] = 'depth of the base of the vertical layer over which statistics are aggregated'
ds_copy.attrs['contact'] = 'service-phy@model-mfc.eu'
ds_copy.attrs['institution'] = "Centro Euro-Mediterraneo sui Cambiamenti Climatici - CMCC, Italy "
ds_copy.attrs['product'] = "BLKSEA_ANALYSISFORECAST_PHY_007_001"
print (ds_copy)

ds_copy.to_netcdf(outname,
                        encoding={'time': {'dtype': 'f4', 'units': units, 'calendar': calendar, '_FillValue': None},
                                  'depths': {'dtype': 'f4', '_FillValue': None},
                                  'forecasts': {'dtype': 'f4', '_FillValue': None},
                                  'stats_temperature': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_salinity': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_sla': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'stats_sst': {'dtype': 'f4', '_FillValue': 1e+20},
                                  'area_names': {'dtype': 'S25','char_dim_name':'string_length'},
                                  'metric_names': {'dtype': 'S25','char_dim_name':'string_length'}}, unlimited_dims='time')
