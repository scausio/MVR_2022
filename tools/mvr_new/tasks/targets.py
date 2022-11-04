import luigi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from re import split
from luigi.task import flatten
from glob import glob
from dataset import Dataset, FancyArray
from gsw import z_from_p


COPY_ATTRIBUTES = ['platform_name', 'platform_code', 'platform', 'source']


class output_data(luigi.Config):
    figure_path = luigi.Parameter(default='.')
    figure_format = luigi.Parameter(default='png')
    data_path = luigi.Parameter(default='.')

    def create_paths(self):
        for path in [self.figure_path, self.data_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

    @property
    def figure_format_list(self):
        return split('[ ,;]+', self.figure_format)


def expand_wildcards(filename):
    if any(c in filename for c in '*?'):
        return sorted(glob(filename))
    else:
        return filename


def merge_profiles(obs1, obs2):
    # First remove masked values
    if np.ma.is_masked(obs1):
        obs1 = obs1[~obs1.mask]
    if np.ma.is_masked(obs2):
        obs2 = obs2[~obs2.mask]

    if obs1.size == 0:
        return obs2
    elif obs2.size == 0:
        return obs1

    # Concatenate the two profiles and find the order
    values = np.concatenate([obs1.view(np.ndarray), obs2.view(np.ndarray)])
    depth = np.concatenate([obs1['depth'], obs2['depth']])
    order = np.argsort(depth)

    # Build a new FancyArray and return
    metadata = obs1.metadata.copy()
    metadata.set_dimension('depth', depth[order])
    return FancyArray(values[order], metadata=metadata)


class InconsistentDataError(Exception):
    pass


class DatasetTarget(luigi.Target):
    def __init__(self, files, **kwargs):
        if isinstance(files, (str, unicode)):
            files = [files]
        self.files = flatten(map(expand_wildcards, files))
        self.dataset = None
        self.fix_coordinates = kwargs.pop('fix_coordinates', False)
        self.time_offset = kwargs.pop('time_offset', 0)
        self.open_kwargs = kwargs

    @property
    def path(self):
        return self.files

    def exists(self):
        return len(self.files) > 0 and all(os.path.isfile(filename) for filename in self.files)

    def open(self, **kwargs):
        if self.dataset is None:
            open_kwargs = self.open_kwargs.copy()
            open_kwargs.update(kwargs)
            self.dataset = Dataset(self.files, **open_kwargs)

            if self.fix_coordinates:
                nav_lat = self.dataset['nav_lat'][...]
                nav_lon = self.dataset['nav_lon'][...]

                # In a pseudo-regular grid the coordinates are curvilinear but the grid is regular
                if not (nav_lat.min(dim='x') == nav_lat.max(dim='x')).all():
                    raise ValueError('latitude changes as a function of x, this is not a regular grid')
                if not (nav_lon.min(dim='y') == nav_lon.max(dim='y')).all():
                    raise ValueError('longitude changes as a function of y, this is not a regular grid')

                # Overwrite x and y with longitude and latitude
                self.dataset.dimensions['y'] = nav_lat.mean(dim='x').view(np.ndarray)
                self.dataset.dimensions['x'] = nav_lon.mean(dim='y').view(np.ndarray)

                # Replace coordinates for all variables that use them
                for variable in self.dataset.variables.values():
                    dimensions = variable.metadata.dimensions

                    for dim in ('y', 'x'):
                        if dim in dimensions:
                            dimensions[dim] = self.dataset[dim]

                # Point the aliases to the new latitude/longitude
                self.dataset.aliases['latitude'] = 'y'
                self.dataset.aliases['longitude'] = 'x'
                self.dataset.aliases['depth'] = 'deptht'

            if self.time_offset:
                offset = np.timedelta64(self.time_offset, 's')

                # Get the real name of the time dimension
                dim = 'time' if 'time' in self.dataset.dimensions else self.dataset.aliases['time']
                self.dataset.dimensions[dim] += offset

                # Replace coordinates for all variables that use time
                for variable in self.dataset.variables.values():
                    dimensions = variable.metadata.dimensions
                    if dim in dimensions:
                        dimensions[dim] = self.dataset[dim]

        return self.dataset


class InSituDatasetTarget(DatasetTarget):

    def open(self):
        for filename in self.files:
            yield Dataset(filename)

    def iterate(self, name, quality):
        for dataset in self.open():
            size = dataset['time'].size

            if not dataset['latitude'].size == size:
                raise InconsistentDataError('sizes of latitude and time do not match (%d != %d)' % (dataset['latitude'].size, size))

            if not dataset['longitude'].size == size:
                raise InconsistentDataError('sizes of longitude and time do not match (%d != %d)' % (dataset['longitude'].size, size))

            # Resolve the standard_name explicitly, since later we need <name>_QC
            if not name in dataset.variables:
                try:
                    name = dataset.aliases[name]
                except:
                    pass

            profile_data = {}
            for index in range(size):
                try:
                    obs = dataset[name](time=index)
                except KeyError:
                    raise StopIteration

                # Check position_qc, skip entire observation if bad
                position_qc = dataset['POSITION_QC'](POSITION=index)
                if position_qc not in quality:
                    continue

                time_qc = dataset['TIME_QC'](TIME=index)
                if time_qc not in quality:
                    continue

                for dim in ('latitude', 'longitude'):
                    obs[dim] = dataset[dim][index]

                try:
                    # Moorings
                    obs['DEPTH'] = dataset['DEPH'][index]
                    depth_qc = dataset['DEPH_QC'][index]
                except KeyError:
                    # Profiles
                    obs['DEPTH'] = np.abs(z_from_p(dataset['PRES'][index], dataset['latitude'][index]))
                    depth_qc = dataset['PRES_QC'][index]
                
                good = np.zeros(obs.shape, dtype='bool')
                for flag in quality:
                    good |= depth_qc == flag
                obs.mask |= ~good

                try:
                    obs_qc = dataset['%s_QC' % name][index]
                    good = np.zeros(obs.shape, dtype='bool')
                    for flag in quality:
                        good |= obs_qc == flag
                    obs.mask |= ~good
                except KeyError:
                    pass

                obs.metadata.aliases['depth'] = 'DEPTH'

                # Copy these attributes from the file to the observation
                for attr in COPY_ATTRIBUTES:
                    try:
                        obs.metadata.attributes[attr] = dataset.attributes[attr]
                    except KeyError:
                        pass

                # FIXME: key probably should be a tuple, but np.ndarray is not hashable
                key = '%s, %s, %s' % (obs['latitude'], obs['longitude'], obs['time'])
                if not key in profile_data:
                    profile_data[key] = obs[0:0]  # Empty slice

                profile_data[key] = merge_profiles(profile_data[key], obs)

            for obs in profile_data.values():
                yield obs

    def enumerate(self, *args, **kwargs):
        return enumerate(self.iterate(*args, **kwargs))


class SatelliteDatasetTarget(DatasetTarget):

    def open(self):
        for filename in self.files:
            yield Dataset(filename)

    def iterate(self, name):
        for dataset in self.open():
            size = dataset['time'].size

            obs = dataset[name][...]
            obs.metadata.unstructured = True
            for dim in ('latitude', 'longitude'):
                obs[dim] = dataset[dim]

            # Copy these attributes from the file to the observation
            for attr in COPY_ATTRIBUTES:
                try:
                    obs.metadata.attributes[attr] = dataset.attributes[attr]
                except KeyError:
                    pass

            yield obs

    def enumerate(self, *args, **kwargs):
        return enumerate(self.iterate(*args, **kwargs))


def _filename(name, extension, date1=None, date2=None, index=None):
    filename = [name]

    if date1:
        filename += ['_{:%Y%m%d}'.format(date1)]

    if date2:
        filename += ['-{:%Y%m%d}'.format(date2)]

    if index:
        filename += ['_{:d}'.format(index)]

    filename += ['.{:s}'.format(extension)]

    return ''.join(filename)


class OutputDatasetTarget(DatasetTarget):
    def __init__(self, name, date1=None, date2=None, index=None, **kwargs):
        output_data().create_paths()
        path = output_data().data_path
        filename = _filename(name, 'nc', date1, date2, index)
        if not 'format' in kwargs:
            kwargs['format'] = 'NETCDF4_CLASSIC'
        super(OutputDatasetTarget, self).__init__(os.path.join(path, filename), **kwargs)


class FigureTarget(luigi.Target):

    def __init__(self, name, date1=None, date2=None, index=None):
        self.name = name
        self.date1 = date1
        self.date2 = date2
        self.index = index

    @property
    def files(self):
        path = output_data().figure_path
        for fmt in output_data().figure_format_list:
            filename = _filename(self.name, fmt, self.date1, self.date2, self.index)
            yield os.path.join(path, filename)
    path = files  # Required for the InputUpdateMixin

    def exists(self):
        return all(os.path.isfile(filename) for filename in self.files)

    def __enter__(self):
        output_data().create_paths()
        self.fig = plt.figure()
        return self.fig

    def __exit__(self, *args, **kwargs):
        for filename in self.files:
            plt.savefig(filename)
        plt.close()
