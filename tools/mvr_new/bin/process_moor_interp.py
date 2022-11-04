#!/usr/bin/env python
import intake
import xarray as xr
import numpy as np
import logging
from argparse import ArgumentParser


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

logger = logging.getLogger('moor')

parser = ArgumentParser(description='Process MOORING data')
parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
parser.add_argument('-n', '--name', default='moor_tracer', help='dataset name (in catalog)')
parser.add_argument('-s', '--start-date', help='start date')
parser.add_argument('-e', '--end-date', help='end date')
parser.add_argument('-o', '--output', help='output file')

args = parser.parse_args()

logger.info('Opening catalog %s' % args.catalog)
cat = intake.open_catalog(args.catalog)

dataset = cat[args.name]

logger.info('Dataset "%s" contains %d files' % (args.name, len(dataset.files)))

if args.start_date:

    dataset = dataset.subset(ref_date=slice(args.start_date, None))
if args.end_date:
    dataset = dataset.subset(ref_date=slice(None, args.end_date))

logger.info('Using subset of %d files' % len(dataset.files))

moor = dataset.read()
moor = moor.isel(obs=moor.depth.values==0)

depths=np.unique(moor['depth'])

print (moor['temperature'].values)
print (len(moor['temperature_qc'].values))
print (moor['time_qc'].values)
print (moor['salinity_qc'].values)
print (moor['salinity'].values)

#moor = moor.where((moor['time_qc'].astype(float) == 1))#.dropna(dim='obs',how='all')
moor = moor.isel(obs=moor['time_qc'].astype(float) == 1.)
moor['temperature'] = moor['temperature'].where((moor['temperature_qc'].astype(float) == 1.))#.dropna(dim='obs',how='all')
moor['salinity'] = moor['salinity'].where((moor['salinity_qc'].astype(float) == 1.))#.dropna(dim='obs',how='all')

moor = moor.get(['temperature', 'salinity'])

logger.info('Observations passing quality flags: %d' % moor.sizes['obs'])

moor = xr.concat([profile for _, profile in moor.groupby('dc_reference')], dim='obs').sortby('time')
moor = moor.dropna(dim='obs',how='all')
#moor = moor.where(moor['depth'].values >= 0.).dropna(dim='obs',how='all').sortby('time')
#moor = moor.isel(obs=moor['depth'].values >= 0.).sortby('time')

logger.info('Observations after resampling high-resolution profiles: %d' % moor.sizes['obs'])

# Remove attributes and add structure for model values
moor.coords['model'] = np.array([], dtype=str)
for name, variable in moor.variables.items():
    if name not in moor.coords:
        model_variable = np.zeros_like(variable, shape=tuple(moor.sizes.values()))
        moor['model_%s' % name] = xr.DataArray(model_variable, dims=moor.sizes.keys())
moor.attrs = {}

print(moor)
#moor=moor.dropna('obs', how='any')
if args.output:
    logger.info('Writing output dataset to %s' % args.output)
    moor.to_netcdf(args.output)
