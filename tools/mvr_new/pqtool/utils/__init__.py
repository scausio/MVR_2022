import xarray as xr
import numpy as np
import glob, datetime, calendar


def expand_wildcards(filename):
    if any(c in filename for c in '*?'):
        return sorted(glob.glob(filename))
    else:
        return filename


def get_datetime_range(year, month):
    nb_days = calendar.monthrange(year, month)[1]
    return [datetime.datetime(year, month, day) for day in range(1, nb_days + 1)]


def metrics(data):
    result = xr.Dataset()

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            bias = data['model_%s' % name] - var
            result['%s_bias' % name] = bias.mean(dim='obs')
            result['%s_rmse' % name] = np.sqrt((bias ** 2.).mean(dim='obs'))
            result['%s_nobs' % name] = bias.count(dim='obs')
    return result


def unbias(data):
    data['sla'] -= data['sla'].mean(dim='obs')
    data['model_sla'] -= data['model_sla'].mean(dim='obs')
    return data


def unbias_along_track(data):
    return data.groupby('track').map(unbias)


def swap_mdt(data):
    if data['model'] in ['bsfs_v3.2', 'bs-nrt_2.2eof3', 'bs-nrt_2.19eof6.1']:
        data['model_sla'] += data['mdt'] - data['old_mdt']
    return data


def covariance(x, y, dim=None):
    valid_values = x.notnull() & y.notnull()
    valid_count = valid_values.sum(dim)

    demeaned_x = (x - x.mean())
    demeaned_y = (y - y.mean())

    return xr.dot(demeaned_x, demeaned_y, dims=dim) / valid_count


def mvr_metrics(data):
    result = xr.Dataset()

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            if name == 'temperature':
                # Convert from Celsius to Kelvin
                data['model_%s' % name] += 273.15
                data[name] += 273.15
            elif name== 'sla':
                # Convert from m to cm
                data['model_%s' % name].values *= 100
                data[name].values *= 100

            # Filter
            #data['model_%s' % name] = data['model_%s' % name].where(
            #     np.abs(data['model_%s' % name] - data[name]) < 4.)
            #data[name] = data[name].where(
            #     np.abs(data['model_%s' % name] - data[name]) < 4.)

            bias = data['model_%s' % name] - data[name]
            result['mse_%s' % name] = (bias ** 2.).groupby('date').mean(dim='obs')
            result['count_%s' % name] = bias.groupby('date').count(dim='obs')
            result['mean_obs_%s' % name] = data[name].groupby('date').mean(dim='obs')
            result['mean_model_%s' % name] = data['model_%s' % name].groupby('date').mean(dim='obs')
            result['var_obs_%s' % name] = data[name].groupby('date').var(dim='obs')
            result['var_model_%s' % name] = data['model_%s' % name].groupby('date').var(dim='obs')
            try:
                result['cov_%s' % name] = xr.cov(data['model_%s' % name], data[name], dim=['latitude', 'longitude'])
            except:
                x = data['model_%s' % name].groupby('date').mean(dim='obs')
                y = data[name].groupby('date').mean(dim='obs')
                result['cov_%s' % name] = covariance(x, y, 'model')
    return result


def mvr_argo_metrics(data):
    result = xr.Dataset()
    # data.coords['time'] = data.time.dt.floor('1D')

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            if name == 'temperature':
                # Filter
                # data['model_%s' % name] = data['model_%s' % name].where(
                #     np.abs(data['model_%s' % name] - data[name]) < 4.)
                # data[name] = data[name].where(
                #     np.abs(data['model_%s' % name] - data[name]) < 4.)
                # var = var.where(
                #     np.abs(data['model_%s' % name] - data[name]) < 4.)

                # Convert from Celsius to Kelvin
                data['model_%s' % name] += 273.15
                # bug fixing, 2 times conversion for obs from C to K
                data[name] += 273.15
                #var += 273.15
            bias = data['model_%s' % name] - data[name]
            result['mse_%s' % name] = (bias ** 2.).groupby('date').mean(dim='obs')
            result['count_%s' % name] = bias.groupby('date').count(dim='obs')
            result['mean_obs_%s' % name] = data[name].groupby('date').mean(dim='obs')
            result['mean_model_%s' % name] = data['model_%s' % name].groupby('date').mean(dim='obs')
            result['var_obs_%s' % name] = data[name].groupby('date').var(dim='obs')
            result['var_model_%s' % name] = data['model_%s' % name].groupby('date').var(dim='obs')

            #x = data['model_%s' % name].groupby('date').mean(dim='obs')
            #y = data[name].groupby('date').mean(dim='obs')
            #result['cov_%s' % name] = covariance(x, y, 'model')
            result['cov_%s' % name] = xr.cov(data['model_%s' % name], data[name], dim=['latitude', 'longitude'])
    return result


def mvr_sla_metrics(data):
    result = xr.Dataset()

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            # Convert from m to cm
            data['model_%s' % name].values *= 100
            data[name].values*=100

            #var *= 100 #bug fix

            bias = data['model_%s' % name] - data[name]

            result['mse_sla'] = (bias ** 2.).mean(dim='obs')
            result['count_sla'] = bias.count(dim='obs')
            result['mean_obs_sla'] = data[name].mean(dim='obs')
            result['mean_model_sla'] = data['model_%s' % name].mean(dim='obs')
            result['var_obs_sla'] = data[name].var(dim='obs')
            result['var_model_sla'] = data['model_%s' % name].var(dim='obs')
            #x = data['model_%s' % name]
            #y = data[name]
            #result['cov_sla'] = covariance(x, y, 'obs')
            result['cov_sla'] = xr.cov(data['model_%s' % name], data[name], dim=['latitude', 'longitude'])

    return result


def mvr_sst_metrics(data):
    result = xr.Dataset()

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            data['model_%s' % name].data[data['model_%s' % name].data == 0] = np.nan
            data['model_%s' % name] = data['model_%s' % name].where(~var.isnull())
            if name == 'temperature':
                # Filter
                data['model_%s' % name] = data['model_%s' % name].where(
                    np.abs(data['model_%s' % name] - data[name]) < 4.)
                data[name] = data[name].where(
                    np.abs(data['model_%s' % name] - data[name]) < 4.)
                var = var.where(
                    np.abs(data['model_%s' % name] - data[name]) < 4.)
                # Convert from Celsius to Kelvin
                data['model_%s' % name] += 273.15
                var += 273.15
                #data[name] += 273.15 #bug fix
            bias = data['model_%s' % name] - var
            result['mse_sst'] = (bias ** 2.).mean(dim=['latitude', 'longitude'])
            result['count_sst'] = bias.count(dim=['latitude', 'longitude'])
            result['mean_obs_sst'] = var.mean(dim=['latitude', 'longitude'])
            result['mean_model_sst'] = data['model_%s' % name].mean(dim=['latitude', 'longitude'])
            result['var_obs_sst'] = var.var(dim=['latitude', 'longitude'])
            result['var_model_sst'] = data['model_%s' % name].var(dim=['latitude', 'longitude'])
            result['cov_sst'] = xr.cov(data['model_%s' % name], data[name], dim=['latitude', 'longitude'])
    return result
