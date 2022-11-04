# -*- coding: utf-8 -*-
import luigi
import mercator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from luigi.task import flatten
from datetime import date, datetime, timedelta
from seaoverland import seaoverland
from re import split

from dataset import Dataset, InterpolationError
from statistics import Statistic, TimeSeries
from tools.mvr_new.tasks.mario import InputUpdateMixin
from tools.mvr_new.tasks.targets import *
from tools.mvr_new.tasks.external import *
from tools.mvr_new.tasks.visualisation import visualisation
from tools.mvr_new.tasks.util import date_range, group_dates, lat_formatter, lon_formatter


class metrics(luigi.Config):
    layers = luigi.Parameter(default='5,10,20,50,100,500,1000')
    insitu_quality = luigi.Parameter(default='1,2')
    sst_quality = luigi.Parameter(default='4')
    sst_foundation_depth = luigi.FloatParameter(default=5.)
    mdt = luigi.Parameter(default='MDT.nc')
    mesh_mask = luigi.Parameter(default='mesh_mask.nc')

    @property
    def layers_array(self):
        return np.array(split('[ ,;]+', self.layers)).astype(float)
        
    @property
    def insitu_quality_array(self):
        return np.array(split('[ ,;]+', self.insitu_quality)).astype(int)

    @property
    def sst_quality_array(self):
        return np.array(split('[ ,;]+', self.sst_quality)).astype(int)


class ProfileComparison(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    maximum_difference = luigi.FloatParameter(default=None)
    model_levels = luigi.Parameter(default=None)
    metric_id = None

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    @property
    def model_levels_array(self):
        levels = map(float, split('[ ,;]+', self.model_levels))
        return np.array(levels)

    def requires(self):
        return [ModelData(subset=self.subset, start_date=self.date),
                ProfileObservations(start_date=self.date)]

    def output(self):
        return OutputDatasetTarget(self.metric_id, self.date)

    def run(self):
        model_task, observation_task = self.requires()

        model = model_task.output().open()
        variable = model[self.model_variable]
        timeseries = TimeSeries(bins=metrics().layers_array)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            variable = variable(**region)

        index = 1
        for observation in observation_task.output().iterate(self.observation_variable,
                                                             metrics().insitu_quality_array):
            if observation['time'] < np.datetime64(self.date):
                continue
            if observation['time'] >= np.datetime64(self.date) + np.timedelta64(1, 'D'):
                continue

            try:
                # Interpolate the model to the location of the observation
                profile = variable.interpolate(time=observation['time'],
                                               latitude=observation['latitude'],
                                               longitude=observation['longitude'])

                if self.model_levels:
                    profile = profile.interpolate(depth=self.model_levels_array)

                diff = profile - observation

                if self.maximum_difference:
                    diff = np.ma.masked_outside(diff, -self.maximum_difference, self.maximum_difference)

            except InterpolationError:
                continue

            if np.ma.is_masked(diff):
                if diff.mask.all():
                    # Skip the profile if all values are masked (i.e. there was no data to compare)
                    continue

            with FigureTarget(self.metric_id, self.date, index=index) as fig:
                plt.plot(observation, observation['depth'], 'o-', label='%s (%s)' % (observation.metadata.attributes.get('platform_name', '-'),
                                                                                     observation.metadata.attributes.get('platform_code', '-')))
                plt.plot(profile, profile['depth'], 'v-', label='Model (%s)' % model.attributes.get('source', '-'))
                blank = Rectangle((0, 0), 1, 1, fc='w', fill=False, edgecolor=None, linewidth=0)
                plt.title('%s,  %s  %s' % (str(observation['time']).replace('T', '  '),
                                           lat_formatter(observation['latitude']),
                                           lon_formatter(observation['longitude'])))
                plt.gca().invert_yaxis()
                plt.yscale('log', subsy=[0])
                plt.gca().set_yticks(metrics().layers_array)
                plt.gca().set_yticklabels(metrics().layers_array)
                plt.gca().yaxis.grid(True, c='silver', ls=':')
                plt.xlabel('%s [%s]' % (self.name, self.units))
                plt.ylabel('Depth [m]')

                handles, labels = plt.gca().get_legend_handles_labels()
                handles += [blank, blank]
                labels += ['BIAS: %g' % diff.mean(),
                           'RMSD: %g' % np.sqrt((diff ** 2.).mean())]
                plt.legend(handles, labels, loc='best')

                index += 1

            metric = timeseries[observation['time']]
            metric.add(diff, diff['depth'])

        timeseries.write(self.output().open(mode='w', format='NETCDF4_CLASSIC'))


class TemperatureProfileComparison(ProfileComparison):
    metric_id = 'T-CLASS2-PROF'
    subset = 'TEMP'
    name = 'Temperature'
    units = u'°C'
    model_variable = input_data().sea_water_temperature
    observation_variable = 'TEMP'


class SalinityProfileComparison(ProfileComparison):
    metric_id = 'S-CLASS2-PROF'
    subset = 'PSAL'
    name = 'Salinity'
    units = '1e-3'
    model_variable = input_data().sea_water_salinity
    observation_variable = 'PSAL'


class MetricTimeSeries(InputUpdateMixin, luigi.Task):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=None)
    metric = luigi.Parameter(default='RMSD')
    frequency = luigi.ChoiceParameter(default='D', choices=['D', 'W', 'M', 'Y'])
    layer = luigi.IntParameter(default=0)
    metric_id = None
    base = None
    hide_bias = False

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - (self.end_date or self.start_date) < timedelta(days=30)

    def requires(self):
        dates = list(group_dates(self.start_date, self.end_date or self.start_date))

        if self.end_date is None or self.start_date == self.end_date:
            # Single day, no grouping
            yield self.base(date=start_date)
        else:
            # Group dates and recursively require MetricTimeSeries for smaller chunks
            for start_date, end_date in group_dates(self.start_date, self.end_date or self.start_date):
                if start_date == end_date:
                    yield self.base(date=start_date)
                else:
                    yield self.__class__(start_date=start_date, end_date=end_date,
                                         frequency=self.frequency, layer=self.layer)

    def output(self):
        return [OutputDatasetTarget(self.metric_id, self.start_date, self.end_date or self.start_date),
                FigureTarget(self.metric_id, self.start_date, self.end_date or self.start_date)]

    @property
    def nobs_bar_width(self):
        widths = {'D': 1, 'W': 7, 'M': 31, 'Y': 366}
        return widths[self.frequency]

    def run(self):
        # All outputs of required tasks
        outputs = flatten([task.output() for task in self.requires()])

        # Use only the datasets as input
        inputs = filter(lambda x: isinstance(x, DatasetTarget), outputs)  # TODO: special dataset class for error stats
        filenames = flatten(map(lambda target: target.files, inputs))
        dataset = Dataset(filenames)
        timeseries = TimeSeries(dataset)

        if self.frequency != 'D':
            timeseries = timeseries.change_units('datetime64[%s]' % self.frequency)

        # Save an aggregated TimeSeries file
        timeseries.write(self.output()[0].open(mode='w', format='NETCDF4_CLASSIC'))

        with self.output()[1] as fig:
            dates, bias, rmsd, count = timeseries.get_metrics('bias', 'rmsd', 'count')

            if not any([var is None for var in [dates, bias, rmsd, count]]):
                ax1 = plt.axes()

                bias = np.ma.array(bias, mask=(count == 0))
                rmsd = np.ma.array(rmsd, mask=(count == 0))

                if not self.hide_bias:
                    ax1.plot(dates.astype(datetime), bias[:,self.layer], 'r.-', label='Bias')
                ax1.plot(dates.astype(datetime), rmsd[:,self.layer], 'b.-', label='RMSD')
                ax1.set_ylabel(u'%s [%s]' % ('Metric', self.base.units))

                ax2 = ax1.twinx()
                ax2.bar(dates.astype(datetime), count[:,self.layer], color='gainsboro', width=self.nobs_bar_width, label='NOBS')
                ax2.set_ylabel('Number of observations (NOBS)')

                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(handles1+handles2, labels1+labels2, loc='best')

                ax1.set_zorder(ax2.get_zorder()+1)  # bars should be behind the metrics
                ax2.patch.set_visible(True)
                ax1.patch.set_visible(False)

                fig.autofmt_xdate()
                plt.subplots_adjust(right=0.85)
                ax1.grid(True, linestyle=':', color='silver')

                bins = timeseries.bins  # FIXME: use the luigi config as the source here?
                if len(bins) > 2:
                    title = '%s at %gm-%gm' % (self.base.name, bins[self.layer], bins[self.layer+1])
                else:
                    title = self.base.name

                plt.title(title)


class SatelliteSSTComparison(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    maximum_difference = luigi.FloatParameter(default=None)
    metric_id = 'SST-D-CLASS4-IRAD-BIAS-T'
    units = u'°C'
    name = 'Sea Surface Temperature'

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    def requires(self):
        return [ModelData(subset='TEMP', start_date=self.date),
                SatelliteSSTObservation(date=self.date)]

    def output(self):
        return OutputDatasetTarget(self.metric_id, self.date)

    def run(self):
        timeseries = TimeSeries()
        timeseries[self.date]  # Create empty entry

        if input_data().ignore_missing == False or self.requires()[1].output().exists():
            model, satellite = [task.output().open() for task in self.requires()]

            # TODO: time selection and depth interpolation could be improved
            depth = metrics().sst_foundation_depth
            observation = satellite['sea_surface_foundation_temperature'](time=0) - 273.15

            # Quality control, points that fail are masked
            good = np.zeros(observation.shape, dtype='bool')
            quality = satellite['quality_level'](time=0)
            for flag in metrics().sst_quality_array:
                good |= quality == flag
            observation.mask |= ~good

            sst = model[input_data().sea_water_temperature].interpolate(depth=depth, time=observation['time'])

            # Selecting only a region of the input data
            region = input_data().region
            if region:
                sst = sst(**region)

            # HACK for AIFS-style data -- EJ 3/9/2018
            #sst[sst == 0.] = np.ma.masked
            #from dataset import regrid
            #observation.metadata.aliases['longitude'] = 'lon'
            #observation.metadata.aliases['latitude'] = 'lat'
            #sst, observation = regrid(sst, observation, dimensions=('latitude', 'longitude'))
            ###

            diff = sst - observation

            if self.maximum_difference:
                diff = np.ma.masked_outside(diff, -self.maximum_difference, self.maximum_difference)


            metric = timeseries[observation['time']]
            metric.add(diff.ravel(), np.ones_like(diff.ravel()) * diff['depth'])

            with FigureTarget(self.metric_id, self.date) as fig:
                vmax = np.abs(diff).max()
                seaoverland(diff, 1)

                ax = plt.axes(projection='mercator')
                ax.coastline(visualisation().coastline, sea=None, zorder=2)
                plt.contourf(diff['longitude'], diff['latitude'],
                             diff, 20, cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)

                plt.colorbar(label=u'%s (model-obs) [%s]' % (self.name, self.units), orientation='horizontal', fraction=0.08, pad=0.08)
                plt.grid(True, linestyle=':', color='black', alpha=0.5)
                plt.title(str(diff['time']).replace('T', '  '))

                xbound = ax.get_xbound()
                ybound = ax.get_ybound()

            with FigureTarget('%s-MODEL' % self.metric_id, self.date) as fig:
                seaoverland(sst, 1)

                ax = plt.axes(projection='mercator')
                ax.coastline(visualisation().coastline, sea=None, zorder=2)
                plt.contourf(sst['longitude'], sst['latitude'],
                             sst, 20, cmap='RdYlBu_r')

                plt.colorbar(label=u'%s (model) [%s]' % (self.name, self.units), orientation='horizontal', fraction=0.08, pad=0.08)
                plt.grid(True, linestyle=':', color='black', alpha=0.5)
                plt.title(str(sst['time']).replace('T', '  '))

                # Make sure axes ranges are the same
                ax.set_xbound(*xbound)
                ax.set_ybound(*ybound)

            with FigureTarget('%s-SATELLITE' % self.metric_id, self.date) as fig:
                seaoverland(observation, 1)

                ax = plt.axes(projection='mercator')
                ax.coastline(visualisation().coastline, sea=None, zorder=2)
                # Here use lon/lat since longitude/latitude is defined multiple times for SST data
                plt.contourf(observation['lon'], observation['lat'],
                             observation, 20, cmap='RdYlBu_r')

                plt.colorbar(label=u'%s (satellite) [%s]' % (self.name, self.units), orientation='horizontal', fraction=0.08, pad=0.08)
                plt.grid(True, linestyle=':', color='black', alpha=0.5)
                plt.title(str(observation['time']).replace('T', '  '))

                # Make sure axes ranges are the same
                ax.set_xbound(*xbound)
                ax.set_ybound(*ybound)

        timeseries.write(self.output().open(mode='w', format='NETCDF4_CLASSIC'))


class SatelliteSLAComparison(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    maximum_difference = luigi.FloatParameter(default=None)
    metric_id = 'SLA-D-CLASS4-ALT-BIAS-T'
    name = 'Sea Level Anomaly'
    units = 'm'

    @property
    def check_input_update(self):
        # Check input files for changes for up to 60 days
        return datetime.now().date() - self.date < timedelta(days=60)

    def requires(self):
        return [ModelData(subset='ASLV', start_date=self.date),
                SatelliteSLAObservation(date=self.date)]

    def output(self):
        return OutputDatasetTarget(self.metric_id, self.date)

    def run(self):
        timeseries = TimeSeries()
        timeseries[self.date]  # Create empty entry

        if input_data().ignore_missing == False or self.requires()[1].output().exists():
            model_task, observation_task = self.requires()

            model = model_task.output().open()
            ssh = model[input_data().sea_surface_height]

            mdt_data = Dataset(metrics().mdt)
            mdt = mdt_data['mdt']

            for index, observation in observation_task.output().enumerate('sea_surface_height_above_sea_level'):

                try:
                    # TODO: ssh2 is a workaround for working around the bad valid_min and valid_max in the BS-MFC files
                    ssh.data.set_auto_mask(False)
                    ssh2 = ssh.interpolate(time=observation['time'])
                    ssh2.mask = ssh2 == ssh2.metadata.attributes['_FillValue']

                    # HACK for AIFS-style data -- EJ 3/9/2018
                    #ssh2.mask = np.logical_or(ssh2.mask, ssh2 == 0.)

                    # Selecting only a region of the input data
                    region = input_data().region
                    if region:
                        ssh2 = ssh2(**region)

                    # TODO: this is a quick fix for interpolating to out of range grid points
                    valid = reduce(np.logical_and, [observation['time'] >= ssh2['time'].min(), observation['time'] <= ssh2['time'].max(),
                                                    observation['latitude'] >= ssh2['latitude'].min(), observation['latitude'] < ssh2['latitude'].max(),
                                                    observation['longitude'] >= ssh2['longitude'].min(), observation['longitude'] < ssh2['longitude'].max()])

                    ssht = ssh2.interpolate(time=observation['time'][valid],
                                            latitude=observation['latitude'][valid],
                                            longitude=observation['longitude'][valid]).diagonal_points()

                    mdtt = mdt.interpolate(latitude=observation['latitude'][valid],
                                           longitude=observation['longitude'][valid]).diagonal_points()

                    ssht -= mdtt

                except InterpolationError:
                    continue
                except ValueError as err:
                    continue

                ssht -= ssht.mean()
                observation -= observation.mean()

                # TODO: this does not yet work for unstructured arrays
                #diff = ssht - observation
                diff = ssht.copy()
                diff -= observation[valid]

                if self.maximum_difference:
                    diff = np.ma.masked_outside(diff, -self.maximum_difference, self.maximum_difference)

                timeseries[observation['time'][0]].add(diff, np.zeros_like(diff), unbiased=True)

                with FigureTarget(self.metric_id, self.date, index=index) as fig:
                    ax = plt.axes(projection='mercator')
                    ax.coastline(visualisation().coastline, sea=None, zorder=2)

                    vmax = np.ma.abs(diff).max()
                    plt.scatter(diff['longitude'], diff['latitude'], c=diff, marker='.', cmap='coolwarm', vmin=-vmax, vmax=vmax)
                    plt.colorbar(label=u'%s (model-obs) [%s]' % (self.name, self.units), orientation='horizontal', fraction=0.08, pad=0.08)
                    plt.grid(True, linestyle=':', color='black', alpha=0.5)
                    plt.title('%s  %s' % (observation.metadata.attributes['platform'], str(diff['time'][0]).replace('T', '  ')))

        timeseries.write(self.output().open(mode='w', format='NETCDF4_CLASSIC'))


class TemperatureProfileMetrics(MetricTimeSeries):
    base = TemperatureProfileComparison

    @property
    def metric_id(self):
        depth0, depth1 = metrics().layers_array[self.layer:self.layer+2]
        return 'T-%.0fm-%.0fm-%s-CLASS4-PROF-%s-XY' % (depth0, depth1, self.frequency, self.metric)


class SalinityProfileMetrics(MetricTimeSeries):
    base = SalinityProfileComparison

    @property
    def metric_id(self):
        depth0, depth1 = metrics().layers_array[self.layer:self.layer+2]
        return 'S-%.0fm-%.0fm-%s-CLASS4-PROF-%s-XY' % (depth0, depth1, self.frequency, self.metric)


class SatelliteSSTMetrics(MetricTimeSeries):
    base = SatelliteSSTComparison

    @property
    def metric_id(self):
        return 'SST-%s-CLASS4-IRAD-%s-XY' % (self.frequency, self.metric)


class SatelliteSLAMetrics(MetricTimeSeries):
    base = SatelliteSLAComparison
    hide_bias = True  # Don't show bias, since it is 0 by definition for SLA

    @property
    def check_input_update(self):
        # Check input files for changes for up to 60 days
        return datetime.now().date() - self.date < timedelta(days=60)

    @property
    def metric_id(self):
        return 'SLA-%s-CLASS4-ALT-%s-XY' % (self.frequency, self.metric)

