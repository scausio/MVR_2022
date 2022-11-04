import numpy as np
import matplotlib.pyplot as plt
import yaml
from munch import Munch

def getConfigurationByID(conf,confId):
    globalConf = yaml.safe_load(open(conf))
    return Munch.fromDict(globalConf[confId])
def castToList(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def preprocess(ds):
    del ds['LONGITUDE']
    del ds['LATITUDE']
    del ds['POSITION_QC']
    return ds

def plot_debug(obs_var, bs_var, var_name, plat_name, figure):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)
    obs_var.plot(color='lightgreen', marker='o', alpha=0.3, label='EUXRo01')
    bs_var.plot(color='blue', label='eas5')
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(xlabel='time (day)', fontsize=12)
    ax.set_ylabel(ylabel='%s' % var_name, fontsize=12)
    ax.legend()
    ax.set_title('Platform: %s' % plat_name, fontsize=12)
    plt.savefig(figure)

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
    var[var==0]=np.nan
    sea_points = np.isfinite(var).flatten()
    return points[sea_points]

def getValidNearest(points, point):
    sea_ix = index_query(points, point[0], point[1])
    print (f'asking for {point}, getting { points[sea_ix][0]}')
    return points[sea_ix][0]
