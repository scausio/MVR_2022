import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

fig_name = '/Users/lstefanizzi/PycharmProjects/mvr_from_pq2/files/output/model-mod/{metric}_{var}_{depth}_2020.png'

layers = {0: '2-5m', 1: '5-10m', 2: '10-20m', 3: '20-30m', 4: '30-50m', 5: '50-75m', 6: '75-100m', 7: '100-200m',
          8: '200-500m', 9: '500-1000m'}

fcs = {0: 'an', 1: 'fc1', 2: 'fc3', 3: 'fc5'}

ylim = {'stats_temperature': 120, 'stats_salinity': 120, 'stats_sst': 12000, 'stats_sla': 600}

if __name__ == '__main__':

    inFile = '/Users/lstefanizzi/PycharmProjects/mvr_from_pq2/files/output/model-mod/product_quality_stats_BLKSEA_ANALYSISFORECAST_PHY_007_001_20200101_20201231.nc'
    #inds = xr.open_dataset(inFile)
    inds = nc.Dataset(inFile)

    times = inds['time']
    time_units = times.units
    #time_units = 'days since 1950-01-01 00:00:00'
    time_cal = 'standard'
    dates = nc.num2date(times[:], units=time_units, calendar=time_cal, only_use_cftime_datetimes=False)

    metrics = ["count", "mean_model", "mean_obs", "mse", "var_model", "var_obs", "cov"]

    for var in ['stats_salinity', 'stats_temperature', 'stats_sst', 'stats_sla']:
        variable = inds[var]
        depths = variable.shape[2]

        for depth in range(depths):
            fig = plt.figure()
            ax = plt.axes()

            for metric in ["bias", "rmse"]:

                for fc in range(1):

                    count = variable[:, fc, depth, 0, 0]

                    if np.all(count == 0):
                        continue

                    model = variable[:, fc, depth, 1, 0]
                    obs = variable[:, fc, depth, 2, 0]
                    md = np.ma.filled(model, np.NaN)
                    obs = np.ma.filled(obs, np.NaN)

                    bias = model - obs
                    # bias = obs - model

                    #bias = np.ma.array(bias, mask=count == 0.)

                    # bias = np.ma.masked_where(count == 0, bias)
                    bias = np.ma.masked_invalid(bias)

                    mse = variable[:, fc, depth, 3, 0]
                    rmse = np.sqrt(mse)
                    #rmse = np.ma.array(rmsd, mask= count == 0.)
                    # rmse = np.ma.masked_where(count == 0, rmsd)
                    rmse = np.ma.masked_invalid(rmse)

                    metrics = {'bias' : bias, "rmse": rmse}
                    #metrics = {'rmse': rmse}
                    metrics[metric][metrics[metric].mask] = np.NaN
                    if var == "stats_temperature":
                        print("here")

                    # variable = np.ma.masked_where(var.data == 0, var)
                    ax.plot(dates, metrics[metric].data, '.-', linewidth=1,
                            label=fcs[fc] +"-" + metric + ' : ' + str(np.round(np.nanmean(metrics[metric].data), 2)))

                    # if fcs[fc] == 'an':
                    #      ax2 = fig.add_subplot(111, sharex=ax, frameon=False)
                    #      ax2.yaxis.tick_right()
                    #      ax2.yaxis.set_label_position("right")
                    #      ax2.set_ylabel('Number of measurements')
                    #      ax2.fill_between(dates, count, color="gray", alpha=0.4)
                    #      ax2.set_ylim([0, ylim[var]])
                    #      ax2.yaxis.set_ticks(np.arange(0, ylim[var] + 0.1, ylim[var]/4))

            ax.legend()
            ax.set(xlabel='Date(months)', ylabel=var + '[' + str(variable.getncattr('units')) + ']')
            plt.title(layers[depth])
            plt.grid(True, linestyle=':', color='silver')
            fig.autofmt_xdate()
            plt.savefig(fig_name.format(metric=metric, var=var, depth=layers[depth]))
            plt.close()


