#!/usr/bin/env python
import intake
import xarray as xr
import numpy as np
import logging
from argparse import ArgumentParser
from pqtool.filters.argo import resample_high_resolution


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

logger = logging.getLogger('argo')

parser = ArgumentParser(description='Process ARGO data')
parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
parser.add_argument('-n', '--name', default='argo', help='dataset name (in catalog)')
parser.add_argument('-s', '--start-date', help='start date')
parser.add_argument('-e', '--end-date', help='end date')
parser.add_argument('-o', '--output', help='output file')

args = parser.parse_args()

logger.info('Opening catalog %s' % args.catalog)
cat = intake.open_catalog(args.catalog)

dataset = cat[args.name]
logger.info('Dataset "%s" contains %d files' % (args.name, len(dataset.files)))

if args.start_date:
    dataset = dataset.subset(date=slice(args.start_date, None))
if args.end_date:
    dataset = dataset.subset(date=slice(None, args.end_date))
logger.info('Using subset of %d files' % len(dataset.files))

argo = dataset.read()

argo = argo.where((argo['pressure_qc'] == 1) &
                  (argo['time_qc'] == 1) &
                  (argo['temperature_qc'] == 1) &
                  (argo['salinity_qc'] == 1)).dropna(dim='obs')
argo = argo.get(['temperature', 'salinity'])
logger.info('Observations passing quality flags: %d' % argo.sizes['obs'])

argo = xr.concat([resample_high_resolution(profile) for _, profile in argo.groupby('dc_reference')], dim='obs')
argo = argo.dropna(dim='obs').sortby('time')
#argo = argo.where(argo['depth'] > 5.).dropna(dim='obs').sortby('time')
logger.info('Observations after resampling high-resolution profiles: %d' % argo.sizes['obs'])

# Remove attributes and add structure for model values
argo.coords['model'] = np.array([], dtype=str)
for name, variable in argo.variables.items():
    if name not in argo.coords:
        model_variable = np.zeros_like(variable, shape=tuple(argo.sizes.values()))
        argo['model_%s' % name] = xr.DataArray(model_variable, dims=argo.sizes.keys())
argo.attrs = {}

print(argo)

if args.output:
    logger.info('Writing output dataset to %s' % args.output)
    argo.to_netcdf(args.output)
