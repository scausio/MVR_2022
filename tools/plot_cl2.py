import os.path
import matplotlib
matplotlib.use('TkAgg')
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import time


def tracers():
    for plat in stats:
        print(base.format(stat=plat, year=date[:4], month=date[4:], var=var))
        fs = natsorted(glob(base.format(stat=plat, year=date[:4], month=date[4:], var=var)))

        # ds = xr.open_mfdataset(fs, concat_dim='time', combine='nested')
        for f in fs:
            print(f)

            fig, ax = plt.subplots()
            ds = xr.open_dataset(f)
            print(ds, 888888)

            try:
                ds = ds.isel(depth=0)
            except:
                continue
            for fc in ds.forecasts:
                try:
                    ds.thetao.sel(forecasts=fc).plot(label=fc.values)
                except:
                    ds.so.sel(forecasts=fc).plot(label=fc.values)
            plt.xticks(rotation=30)
            moor = moors.isel(obs=moors.dc_reference.values == str(plat))  # .isel(obs=moors.time==ds.time)
            # ax.plot(moor.time.values, moor.temperature.values, linestyle='dotted', c='k',lw=0.3, label='moor')
            plt.title(f'Platform {plat}')
            plt.legend()
            plt.show()


def uv2deg(u,v):
    return (90-np.rad2deg(np.arctan2(v,u)))%360

def speed(u,v):
    return np.sqrt(v**2+u**2)

def currents_speed():
    #"Current to direction relative true north" HCDT
    for plat in stats:
        moors = xr.open_dataset(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/moorings/{plat}_202201_202209.nc')
        print (moors)
        print(base.format(stat=plat, year=date[:4], month=date[4:], var=var))
        fs = natsorted(glob(base.format(stat=plat, year=date[:4], month=date[4:], var=var)))

        # ds = xr.open_mfdataset(fs, concat_dim='time', combine='nested')
        for f in fs:
            print(f)

            fig, ax = plt.subplots()
            ds = xr.open_dataset(f)
            u=ds['uo']
            v = ds['vo']

            print(ds, 888888)

            try:
                ds = ds.isel(depth=0)
            except:
                continue
            for fc in ds.forecasts:
                _ds=ds.sel(forecasts=fc)
                curr=speed(_ds.uo,_ds.vo)
                curr.plot(label=fc.values)

            plt.xticks(rotation=30)
             # .isel(obs=moors.time==ds.time)
            moor=moors.sel(TIME=ds.time,method='nearest')

            ax.plot(moor.TIME.values, moor.HCSP, linestyle='dotted', c='k',lw=0.8, label='moor')
            plt.title(f'Platform {plat} Currents_speed')
            plt.legend()
            plt.show()



def currents_speed_wholeTS():
    #"Current to direction relative true north" HCDT
    for plat in stats:
        moors = xr.open_dataset(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/moorings/{plat}_202201_202209.nc')
        print (moors)
        print(base.format(stat=plat, year=date[:4], month=date[4:], var=var))
        #fs = natsorted(glob(base.format(stat=plat, year=date[:4], month=date[4:], var=var)))
        fs = natsorted(glob(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class2/v2/UV/{plat}*{date}*_UV.nc'))
        #fs = natsorted(glob(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class2/v2/*/*/{plat}*{date}*_TEMP.nc'))
        # ds = xr.open_mfdataset(fs, concat_dim='time', combine='nested')

        print (fs)
        fig, ax = plt.subplots()
        ds = xr.open_mfdataset(fs)

        print (ds.time)
        u=ds['uo']
        v = ds['vo']

        print(ds, 888888)

        try:
            ds = ds.isel(depth=1)
        except:
            continue
        for fc in ds.forecasts:
            _ds=ds.sel(forecasts=fc)


            curr=speed(_ds.uo,_ds.vo)
            curr.plot(label=fc.values)
            #plt.plot(curr,label=fc.values)
            print(len(curr))
            plt.xticks(rotation=30)
             # .isel(obs=moors.time==ds.time)
            moor=moors.sel(TIME=ds.time,method='nearest')

            #ax.plot(moor.TIME.values, moor.HCSP, linestyle='dotted', c='k',lw=0.8, label='moor')
            plt.title(f'Platform {plat} Currents_speed')
            plt.legend()
            plt.show()



def currents_dir():
    #"Current to direction relative true north" HCDT
    for plat in stats:
        moors = xr.open_dataset(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/moorings/{plat}_202201_202209.nc')
        print (moors)
        print(base.format(stat=plat, year=date[:4], month=date[4:], var=var))
        fs = natsorted(glob(base.format(stat=plat, year=date[:4], month=date[4:], var=var)))

        # ds = xr.open_mfdataset(fs, concat_dim='time', combine='nested')
        for f in fs:
            print(f)

            fig, ax = plt.subplots()
            ds = xr.open_dataset(f)
            u=ds['uo']
            v = ds['vo']

            print(ds, 888888)

            try:
                ds = ds.isel(depth=0)
            except:
                continue
            for fc in ds.forecasts:
                _ds=ds.sel(forecasts=fc)
                curr=uv2deg(_ds.uo,_ds.vo)
                curr.plot(label=fc.values)

            plt.xticks(rotation=30)
             # .isel(obs=moors.time==ds.time)
            moor=moors.sel(TIME=ds.time,method='nearest')

            ax.plot(moor.TIME.values, moor.HCDT, linestyle='dotted', c='k',lw=0.8, label='moor')
            plt.title(f'Platform {plat} Currents_dir')
            plt.legend()
            plt.show()


stats=['EUXRo01','EUXRo02','EUXRo03']#[15552,15428,15480,15499,15360,15655]#['EUXRo01','EUXRo02','EUXRo03']#},# ,'EUXRo01','EUXRo02','EUXRo03'
date='202*'

# 4 temo and sal
# var='TEMP'
# base='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class2/v2/{year}/{month}/{stat}*_{var}*nc'
# moors=[]
# for moor in natsorted(glob(f'/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/interp/moor_tracer_*.nc')):
#     moors.append(xr.open_dataset(moor))
# moors=xr.concat(moors,dim='obs')
# print (moors)
# tracers()
# exit()
# 4 currents
var='TEMP'
base='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/UV/{stat}*2022*_{var}*nc'
#currents_speed()
#currents_dir()
currents_speed_wholeTS()

#import pickle
#figx = pickle.load(open('test.fig.pickle', 'rb'))
#figx.show()



exit()



'''

for plat in stats:
    fs=natsorted(glob(base.format(stat=plat,date='202201*',var=var)))

    #ds = xr.open_mfdataset(fs, concat_dim='time', combine='nested')
    for f in  fs:
        fig, ax = plt.subplots()
        ds = xr.open_dataset(f).isel(time=range(210))
        print (ds)
        print (ds)
        try:
            ds=ds.isel(depth=0)
        except:
            continue
        for fc in ds.forecasts:
            v=ds.thetao.sel(forecasts=fc).values
            a= v[::24][1:]
            b=v[23:][::24]
            print (ds.time.values[23:][::24])
            print(ds.time.values[::24][1:])
            plt.plot(ds.time.values[23:][::24],a-b)
            #ds.thetao.sel(forecasts=fc).plot(label=fc.values)
            #ds.thetao.sel(forecasts=fc).plot(label=fc.values)
        plt.xticks(rotation=30)
        moor=moors.isel(obs=moors.dc_reference.values==str(plat))
        ax.plot(moor.time.values, moor.temperature.values, linestyle='dotted', c='k',lw=0.3, label='moor')
        plt.title(f'Platform {plat}')
        plt.legend()
        plt.show()
    #import pickle

    #pickle.dump(fig, open('test.fig.pickle', 'wb'))
    #plt.clf()
'''


# plot difference
# da=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/files/20220201_h-CMCC--TEMP-BSeas4-BS-b20220203_an-sv10.00.nc')
# print (da)
# da=da.isel(depth=0).thetao.mean('time')
# print()
# df=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/files/20220201_h-CMCC--TEMP-BSeas4-BS-b20220128_fc-sv10.00.nc').isel(depth=0).thetao.mean('time')

# plt.imshow(da-df,cmap='seismic',vmin=-1,vmax=1)
# plt.show()

# plot interp
'''
fs=glob('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/interp/cl2_moor_tracer_fc_*_202202.nc')
for i,f in enumerate(fs):
    print (f)
    ds=xr.open_dataset(f).isel(model=0)
    print (ds.depth.values)

    #.isel(model=0)
    print (ds.model_temperature.values)
    plt.plot(ds.model_temperature.isel(obs=ds.depth==0),label=i)
plt.legend()
plt.show()
'''
exit()
fs=xr.open_mfdataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/interp/15655_MOD_TS_BSMFC-PU_*_TEMP.nc')
print (fs)
fig,ax = plt.subplots()
for i,f in enumerate(fs):
    print (f)
    ds=xr.open_dataset(f).isel(model=0)
    print (ds.depth.values)

    #.isel(model=0)
    print (ds.model_temperature.values)
    ax.plot(ds.model_temperature.isel(obs=ds.depth==0),label=i)
plt.legend()
#plt.show()


exit()
dm=xr.open_dataset('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/interp/moor_tracer_202202.nc')
#dd=dm.salinity.groupby('dc_reference')
#for i,d in dd:
#    d.plot()
plt.plot(dm.temperature,label='moor')
print (dm)
plt.legend()
plt.show()




fs=glob('/Users/scausio/Dropbox (CMCC)/PycharmData/MVR/interp/cl2_moor_tracer_fc_0_202202.nc')
for f in fs:
    print (f)
    ds=xr.open_dataset(f).isel(depth=0)
    print (ds)

    for fc in ds.forecasts:
        try:
            plt.plot(ds.thetao.sel(forecasts=fc),label=fc.values)
        except:
            plt.plot(ds.so.sel(forecasts=fc),label=fc.values)
plt.legend()
plt.show()