#!/usr/bin/env python
import intake
import xarray as xr
import numpy as np
import logging
from argparse import ArgumentParser


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

logger = logging.getLogger('sst')

parser = ArgumentParser(description='Process SST data')
parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
parser.add_argument('-n', '--name', default='sst', help='dataset name (in catalog)')
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

sst = dataset.read()

#sst = sst.where(sst['temperature_qc'] ==1 )
sst = sst.where(sst['error'] <0.4 )
sst = sst.get(['temperature'])
logger.info('Observations passing quality flags: %d' % sst['temperature'].notnull().sum())

# Convert from Kelvin to Celsius
if sst['temperature'].attrs['units'] == 'kelvin':
    sst['temperature'] -= 273.15
    sst['temperature'].attrs['units'] = 'degrees_C'

# Remove attributes and add structure for model values
sst.coords['model'] = np.array([], dtype=str)
for name, variable in sst.data_vars.items():
    model_variable = np.zeros_like(variable, shape=tuple(sst.sizes.values()))
    sst['model_%s' % name] = xr.DataArray(model_variable, dims=sst.sizes.keys())
sst.attrs = {}

print(sst)

if args.output:
    logger.info('Writing output dataset to %s' % args.output)
    sst.to_netcdf(args.output)
