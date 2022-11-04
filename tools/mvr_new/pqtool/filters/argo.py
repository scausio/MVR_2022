import xarray as xr
import numpy as np


def rolling_average(profile, delta=0.5, variables=['temperature', 'salinity']):
    """
    Filter replacing ARGO observations with a rolling average along the depth
    dimension, smoothing the profile and resolving issues with ARGO that
    measure at too high frequencies.
    """
    profile = profile.sortby('depth')
    profile2 = profile.get(variables)    
    for var in variables:
        profile2['%s_count' % var] = profile2[var].notnull()

    # Build the cumulative sum of the variables and add an extra zero variable
    # to the beginning that allows subtracting nothing below if imin=0
    cumsum = profile2.cumsum(dim='obs')
    cumsum = xr.concat([0.*cumsum.isel(obs=0), cumsum], dim='obs')

    # Search the indexes in depth between which the values need to be averaged
    depth = profile2['depth'].data
    imin = np.searchsorted(depth, depth-delta, side='left')
    imax = np.searchsorted(depth, depth+delta, side='right')
    imin[imin < 0] = 0
    imax[imax > len(depth)] = len(depth)
 
    # Get the sum using cumsum[imax] - cumsum[imin] and divide by the number of
    # observations to obtain the average for each interval
    result = cumsum.isel(obs=imax) - cumsum.isel(obs=imin)
    for var in variables:
        # Use .data to replace the contents without modifying the attributes
        profile[var].data = (result[var] / result['%s_count' % var]).data 

    return profile


def keep_closest(profile, levels):
    """
    ARGO filter that keeps only those observations that are the closest to a
    model level.
    """
    size = len(profile['obs'])
    if size < 2:
        return profile

    # Make depth the primary coordinate
    profile2 = profile.swap_dims({'obs': 'depth'})

    # Add a variable index that numbers the observations along the profile
    profile2['index'] = xr.DataArray(np.arange(size), dims='depth')

    # Find the index closest to each level by nearest-neighbour interpolation
    index = profile2['index'].interp(depth=levels, method='nearest').dropna('depth').astype(int)

    # Create a mask that keeps only observations that were the closest observation to one of the levels
    keep = xr.DataArray(np.zeros(size, dtype='bool'), dims='obs')
    keep[index] = True

    return profile.where(keep)


def keep_threshold(profile, levels, delta=0.5):
    """
    ARGO filter that keeps only observations that are within a certain
    threshold from a model level.
    """
    size = len(profile['obs'])
    keep = np.zeros(size, dtype='bool')

    for level in levels:
        keep = np.logical_or(keep, np.logical_and(profile['depth'] >= level-delta,
                                                  profile['depth'] < level+delta))

    keep = xr.DataArray(keep, dims='obs')
    return profile.where(keep)


def resample_high_resolution(profile, variables=['temperature', 'salinity']):
    """
    ARGO filter that resamples profiles with too high vertical resolution onto
    a regular 1m-resolution grid.
    """
    profile = profile.sortby('depth').load()
    depth = profile.coords['depth'].data

    if len(depth) < 2 or np.diff(depth).min() > 1.:
        # Nothing to be done
        return profile

    profile2 = profile.get(variables)
    for var in variables:
        profile2['%s_count' % var] = profile2[var].notnull()

    # Build the cumulative sum of the variables and add an extra zero variable
    # to the beginning that allows subtracting nothing below if imin=0
    cumsum = profile2.cumsum(dim='obs')
    cumsum = xr.concat([0.*cumsum.isel(obs=0), cumsum], dim='obs')

    # Determine the bins
    dmin = np.floor(depth.min() - 0.5)
    dmax = np.ceil(depth.max() - 0.5)
    bins = np.arange(dmin, dmax + 1) + 0.5

    # Search the indexes in depth between which the values need to be averaged
    imin = np.searchsorted(depth, bins[:-1], side='left')
    imax = np.searchsorted(depth, bins[1:], side='right')
    imin[imin < 0] = 0
    imax[imax > len(depth)] = len(depth)

    # Get the sum using cumsum[imax] - cumsum[imin] and divide by the number of
    # observations to obtain the average for each interval
    result = cumsum.isel(obs=imax) - cumsum.isel(obs=imin)
    result['depth'] = xr.DataArray((bins[1:] + bins[:-1]) / 2., dims='obs')

    # Drop grid levels without data
    count = xr.concat([result['%s_count' % var] for var in variables], dim='_tmp').sum(dim='_tmp')
    result = result.where(count > 0).dropna('obs', how='all')

    # Replace the variables in profile
    profile = profile.isel(obs=slice(0, result.sizes['obs']))
    profile.coords['depth'].data = result['depth'].data
    for var in variables:
        # Use .data to replace the contents without modifying the attributes
        profile[var].data = (result[var] / result['%s_count' % var]).data

    return profile
