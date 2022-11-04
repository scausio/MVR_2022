#!/usr/bin/env python
import xarray as xr
import numpy as np
from pqtool.common import common
from scipy.spatial import cKDTree


def find_time(model, coords, **kwargs):
    obs_values = coords['time'].data
    argmin_times = []
    argmin_differences = []
    argmin_idxs = []
    for obs_time in obs_values:
        obs_time_norm = [obs_time]*len(model.time.data)
        model_times = [np.datetime64(model_time) for model_time in model.time.data]

        hours_differences = [np.abs(model - obs).astype('timedelta64[h]') for obs, model in zip(obs_time_norm, model_times)]
        try:
            argmin = np.argmin(hours_differences)[0]
        except IndexError:
            argmin = np.argmin(hours_differences)
        argmin_times.append(model_times[argmin])
        argmin_differences.append(hours_differences[argmin])
        argmin_idxs.append(argmin)

    return argmin_times, argmin_differences, argmin_idxs


def interp_shyfem(source, coords, **kwargs):

    shyfem_lons = np.asarray(source.longitude)
    shyfem_lats = np.asarray(source.latitude)
    tree = cKDTree(list(zip(shyfem_lons, shyfem_lats)))

    argo_lons = np.asanyarray(coords['longitude'])
    argo_lats = np.asanyarray(coords['latitude'])
    return tree.query(list(zip(argo_lons, argo_lats)))


def restructure_model(source, time_values, time_idxs):
    # Creation of new Dataset
    obs_xarray = xr.DataArray(source.coords["obs"].data, name="obs", dims=["obs"])
    times_xarray = xr.DataArray(time_values, name="time", dims=["obs"])
    lon_xarray = xr.DataArray(source.coords["longitude"].data, name="longitude", dims=["obs"])
    lat_xarray = xr.DataArray(source.coords["latitude"].data, name="latitude", dims=["obs"])
    level_xarray = xr.DataArray(source.coords["model_depth"].data, name="model_depth", dims=["model_depth"])

    temp_array = source.temperature.values[time_idxs, source.coords["obs"].data, :]
    sal_array = source.salinity.values[time_idxs, source.coords["obs"].data, :]

    new_model = xr.Dataset(
        {
            "temperature": (
                ("obs", "model_depth"),
                temp_array,
            ),
            "salinity": (
                ("obs", "model_depth"),
                sal_array,
            )
        },
        coords={"obs": obs_xarray, "time": times_xarray, "longitude": lon_xarray,
                "latitude": lat_xarray, "model_depth": level_xarray},
    )
    return new_model


parser = common.create_parser()

args = parser.parse_args()

intermediate, model = common.preprocess(args.input, args.catalog, args.name,
                                        args.start_date, args.end_date)

# Select nearest in all coordinates except depth
nearest_coords = {k: v for k, v in intermediate.coords.items() if k not in ['model', 'dc_reference']}
dist, indx = interp_shyfem(model, nearest_coords, method='nearest')

# TODO: This should not be created in the driver in the first place. For shyfem we already have level
del model.coords['depth']

model = model.rename({'level': 'model_depth', 'node': 'obs'}).isel(obs=indx)

times, time_differences, times_idx = find_time(model, nearest_coords)

restructured_model = restructure_model(model, times, times_idx)

restructured_model = common.interpolate_over_depth(intermediate, restructured_model)

common.postprocess(intermediate, restructured_model, args.name, args.output)
