#!/usr/bin/env python
import intake
import xarray as xr
import numpy as np
import logging
from argparse import ArgumentParser
from scipy import spatial


def grid2points(lon,lat):
    xx,yy=np.meshgrid(lon,lat)
    return np.array((xx.flatten(), yy.flatten())).T

def index_query(points,x_point,y_point):
    tree = spatial.KDTree(points)
    return tree.query([(x_point, y_point)])[1]


def getValidPoints(var, lon, lat):
    if len(var.shape) == 3:
        var = var[0]
    elif len(var.shape) == 4:
        var = var[0][0]

    points = grid2points(lon, lat)
    var.values[var.values==0]=np.nan
    sea_points = np.isfinite(var.values).flatten()
    return points[sea_points]

def getValidNearest(points, point):
    sea_ix = index_query(points, point[0], point[1])
    print (len(sea_ix))
    return points[sea_ix][0]


def interp(source, coords, **kwargs):
    """
    This is a more efficient version of DataArray.interp() for arrays that are
    not in memory. Where DataArray.interp() loads all the data, this loads only
    the coordinates and the nearest points.
    """
    index = {}
    kwargs['method'] = 'nearest'
    mask = np.zeros_like(len(coords['time']), dtype='bool')
    for dim, values in coords.items():

        tmp = xr.DataArray(np.arange(len(source[dim])), dims=dim)
        tmp.coords[dim] = source[dim]

        ind = tmp.interp({dim: values}, **kwargs)
        index[dim] = ind.fillna(0).astype(int)
        mask = np.logical_or(mask, np.isnan(ind))

    return source.isel(**index).where(~mask).load()


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger('model')

parser = ArgumentParser(description='Interpolate model results')
parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
parser.add_argument('-n', '--name', required=True, help='dataset name (in catalog)')
parser.add_argument('-s', '--start-date', help='start date')
parser.add_argument('-e', '--end-date', help='end date')
parser.add_argument('-i', '--input', required=True, help='input file')
parser.add_argument('-o', '--output', required=True, help='output file')
parser.add_argument('-vc', '--validation_class', required=True, help='Class_4 or Class_2')
args = parser.parse_args()

logger.info('Opening input file %s' % args.input)
intermediate = xr.open_dataset(args.input)
intermediate['model'] = intermediate['model'].astype(str)  # Workaround for bug in xarray

logger.info('Opening catalog %s' % args.catalog)
cat = intake.open_catalog(args.catalog)

dataset = cat[f"{args.validation_class}_{args.name}"]
logger.info('Dataset "%s" contains %d files' % (args.name, len(dataset.files)))

if args.start_date:
    dataset = dataset.subset(date=slice(args.start_date, None))
if args.end_date:
    dataset = dataset.subset(date=slice(None, args.end_date))
logger.info('Using subset of %d files' % len(dataset.files))

#dataset=dataset.subset(var=['TEMP','PSAL'])
model = dataset.read()
print ('depths',np.unique(intermediate.depth.values))
model=model.sel(depth=np.unique(intermediate.depth.values),method='nearest').load()
print (model)
points=getValidPoints(model.temperature, model.longitude,model.latitude)

new_coords=np.array([getValidNearest(points,point) for point in np.array((intermediate.longitude,intermediate.latitude)).T])

intermediate.longitude.values=new_coords[:,0]
intermediate.latitude.values=new_coords[:,1]

# Select nearest in all coordinates except depth
nearest_coords = {k:v for k, v in intermediate.coords.items() if k not in ['model', 'dc_reference']}

model = model.sel( nearest_coords, method='nearest')

model=model.assign_coords(depth=intermediate.coords['depth'],time=intermediate.coords['time'])

model = model.rename({'temperature': 'model_temperature',
                      'salinity': 'model_salinity'})
model.coords['model'] = xr.DataArray(np.array([args.name]), dims='model')
model = xr.concat([model], dim='model')

intermediate = intermediate.merge(model)

if args.output:
    logger.info('Writing output dataset to %s' % args.output)
    intermediate.to_netcdf(args.output)

