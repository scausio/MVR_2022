#!/usr/bin/env python
import intake
import xarray as xr
import numpy as np
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

logger = logging.getLogger('sla')

parser = ArgumentParser(description='Process SLA data')
parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
parser.add_argument('-n', '--name', default='sla', help='dataset name (in catalog)')
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

sla = dataset.read()

# sla = sla.where((sla['pressure_qc'] == 1) &
#                  (sla['time_qc'] == 1) &
#                  (sla['temperature_qc'] == 1) &
#                  (sla['salinity_qc'] == 1)).dropna(dim='obs')
sla = sla.get([var for var in dataset.metadata['variables'].keys()])
logger.info('Observations passing quality flags: %d' % sla.sizes['obs'])

# Remove attributes and add structure for model values
sla.coords['model'] = np.array([], dtype=str)
for name, variable in sla.data_vars.items():
    model_variable = np.zeros_like(variable, shape=tuple(sla.sizes.values()))
    sla['model_%s' % name] = xr.DataArray(model_variable, dims=sla.sizes.keys())
sla.attrs = {}

print(sla)

if args.output:
    logger.info('Writing output dataset to %s' % args.output)
    for var in sla.variables.values():
        remove_attr = ['scale_factor', 'add_offset']
        if any([attr in var.encoding for attr in remove_attr]):
            var.encoding['dtype'] = var.dtype

            for attr in remove_attr:
                if attr in var.encoding:
                    del var.encoding[attr]

    sla.to_netcdf(args.output)
