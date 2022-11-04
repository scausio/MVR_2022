# -*- coding: utf-8 -*-
import luigi
import numpy as np
import mercator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, time, datetime, timedelta
from seaoverland import seaoverland
from re import split

from dataset import Dataset
from .mario import InputUpdateMixin
from .external import ModelData, ProfileObservations, input_data
from .targets import FigureTarget
from .util import lon_formatter


# Approximate conversion degrees->meter for the Black Sea latitude
METER_PER_DEGREE = np.array([110e3, 80e3])


class visualisation(luigi.Config):
    coastline = luigi.Parameter(default='blacksea_h.shp')
    depth_levels = luigi.Parameter(default='3,5,10,20,30,50,75,100,200,500,1000')
    bathymetry = luigi.Parameter(default='bathy_meter.nc')

    @property
    def depth_levels_array(self):
        return np.array(split('[ ,;]+', self.depth_levels)).astype(float)


class ScalarField(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    subset = 'TEMP'
    interpolate_kwargs = {}
    time = time(12)
    cmap = 'viridis'

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    def requires(self):
        return ModelData(subset=self.subset, start_date=self.date, end_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        field = model[self.variable].interpolate(time=time, **self.interpolate_kwargs)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            field = field(**region)

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = '%s [%s]' % (field.metadata.attributes['long_name'].capitalize(),
                                 field.metadata.attributes['units'].replace('degrees_', u'°'))

            plt.contourf(field['longitude'], field['latitude'], field, 20, cmap=self.cmap)
            plt.colorbar(orientation='horizontal', label=label)


class MixedLayerDepth(ScalarField):
    date = luigi.DateParameter()
    subset = 'AMXL'
    metric_id = 'MLD-D-CLASS1-BLKS'
    variable = 'ocean_mixed_layer_thickness_defined_by_sigma_theta'


class SeaSurfaceHeight(ScalarField):
    date = luigi.DateParameter()
    subset = 'ASLV'
    metric_id = 'SSH-D-CLASS1-BLKS'
    variable = input_data().sea_surface_height


class UpwardWaterFlux(ScalarField):
    metric_id = 'WAFLUP-D-CLASS1-BLKS'
    variable = 'water_flux_out_of_sea_ice_and_sea_water'
    cmap = 'viridis'


class ShortwaveRadiation(ScalarField):
    metric_id = 'HFLDO-D-CLASS1-BLKS'
    variable = 'net_downward_shortwave_flux_at_sea_water_surface'
    cmap = 'inferno'


class DownwardHeatFlux(ScalarField):
    metric_id = 'HEFLDO-D-CLASS1-BLKS'
    variable = 'surface_downward_heat_flux_in_sea_water'
    cmap = 'inferno'


class Transect(InputUpdateMixin, luigi.Task):
    location = luigi.Parameter()
    date = luigi.DateParameter()
    subset = 'TEMP'
    time = time(12)
    cmap = 'viridis'
    
    @property
    def interpolate_kwargs(self):
        dir = self.location[-1]
        coord = float(self.location[:-1])

        if dir in ['W', 'S']:
            coord = -coord

        if dir in ['N', 'S']:
            return {'latitude': coord}
        elif dir in ['E', 'W']:
            return {'longitude': coord}
        else:
            raise RuntimeError('transect location should be specified as "30E", "10N", "4.5W" etc.')

    def requires(self):
        return ModelData(subset=self.subset, start_date=self.date, end_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()

        field = model[self.variable].interpolate(time=time, **self.interpolate_kwargs)
        field = np.ma.masked_equal(field, 0)

        if self.location[-1] in ['N', 'S']:
            xaxis = 'longitude'
            formatter = mercator.ticker.DegreeFormatter(labels=['W', 'E'])
        else:
            xaxis = 'latitude'
            formatter = mercator.ticker.DegreeFormatter(labels=['S', 'N'])

        bath_data = Dataset(visualisation().bathymetry)
        bathymetry = bath_data['sea_floor_depth'].interpolate(**self.interpolate_kwargs)
        
        # For logarithmic scale
        bathymetry.data[bathymetry.data < 1e-6] = 1e-6

        with self.output() as fig:
            ax = plt.axes()

            label = '%s [%s]' % (field.metadata.attributes['long_name'].capitalize(),
                                 field.metadata.attributes['units'].replace('degree_', u'°'))

            plt.title(u'Transect at %s°%s' % (self.location[:-1], self.location[-1]))
            plt.contourf(field[xaxis], field['depth'], field, 20, cmap=self.cmap)
            plt.colorbar(orientation='horizontal', label=label)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(formatter)
            plt.ylabel('Depth [m]')
            plt.semilogy()

            plt.autoscale(False)
            plt.fill_between(bathymetry[xaxis], bathymetry, 20e3, color='seashell')
            plt.plot(bathymetry[xaxis], bathymetry, '-', color='gray')


class TemperatureTransect(Transect):
    variable = input_data().sea_water_temperature
    cmap = 'RdYlBu_r'

    @property
    def metric_id(self):
        return 'T-D-SECT-%s-CLASS2-BLKS' % (self.location)


class SalinityTransect(Transect):
    variable = input_data().sea_water_salinity
    cmap = 'PuOr_r'

    @property
    def metric_id(self):
        return 'S-D-SECT-%s-CLASS2-BLKS' % (self.location)


class CurrentsTransect(Transect):
    subset = 'RFVL'
    cmap = 'YlOrRd'

    @property
    def metric_id(self):
        return 'UV-D-SECT-%s-CLASS2-BLKS' % (self.location)

    def run(self):
        # TODO: rewrite without duplicating the run() method
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()

        u = model[input_data().eastward_sea_water_velocity].interpolate(time=time, **self.interpolate_kwargs)
        v = model[input_data().northward_sea_water_velocity].interpolate(time=time, **self.interpolate_kwargs)
        field = np.hypot(u, v)
        field = np.ma.masked_equal(field, 0)

        if self.location[-1] in ['N', 'S']:
            xaxis = 'longitude'
            formatter = mercator.ticker.DegreeFormatter(labels=['W', 'E'])
        else:
            xaxis = 'latitude'
            formatter = mercator.ticker.DegreeFormatter(labels=['S', 'N'])

        bath_data = Dataset(visualisation().bathymetry)
        bathymetry = bath_data['sea_floor_depth'].interpolate(**self.interpolate_kwargs)
        
        # For logarithmic scale
        bathymetry.data[bathymetry.data < 1e-6] = 1e-6

        with self.output() as fig:
            ax = plt.axes()

            label = '%s [%s]' % ('Sea water velocity',  '[m/s]')

            plt.title(u'Transect at %s°%s' % (self.location[:-1], self.location[-1]))
            plt.contourf(u[xaxis], u['depth'], field, 20, cmap=self.cmap)
            plt.colorbar(orientation='horizontal', label=label)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(formatter)
            plt.ylabel('Depth [m]')
            plt.semilogy()

            plt.autoscale(False)
            plt.fill_between(bathymetry[xaxis], bathymetry, 20e3, color='seashell')
            plt.plot(bathymetry[xaxis], bathymetry, '-', color='gray')



class Transects(luigi.WrapperTask):
    date = luigi.DateParameter()
    locations = luigi.Parameter()

    @property
    def locations_list(self):
        return split('[ ,;]+', self.locations)

    def requires(self):
        for location in self.locations_list:
            yield TemperatureTransect(date=self.date, location=location)
            yield SalinityTransect(date=self.date, location=location)
            yield CurrentsTransect(date=self.date, location=location)


class Temperature(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    time = time(12)
    depth = luigi.FloatParameter()

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    @property
    def metric_id(self):
        return 'T-%.0fm-CLASS1-BLKS' % self.depth

    def requires(self):
        return ModelData(subset='TEMP', start_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        temp = model[input_data().sea_water_temperature].interpolate(time=time, depth=self.depth)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            temp = temp(**region)

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = '%s at %gm [%s]' % (temp.metadata.attributes['long_name'].capitalize(),
                                        self.depth,
                                        temp.metadata.attributes['units'].replace('degrees_', u'°'))

            plt.contourf(temp['longitude'], temp['latitude'], temp, 20, cmap='RdYlBu_r')
            plt.colorbar(orientation='horizontal', label=label)


class Salinity(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    time = time(12)
    depth = luigi.FloatParameter()

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    @property
    def metric_id(self):
        return 'S-%.0fm-CLASS1-BLKS' % self.depth

    def requires(self):
        return ModelData(subset='PSAL', start_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        sal = model[input_data().sea_water_salinity].interpolate(time=time, depth=self.depth)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            sal = sal(**region)

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = '%s at %gm [%s]' % (sal.metadata.attributes['long_name'].capitalize(),
                                        self.depth,
                                        sal.metadata.attributes['units'])

            plt.contourf(sal['longitude'], sal['latitude'], sal, 20, cmap='PuOr_r')
            plt.colorbar(orientation='horizontal', label=label)


class Currents(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    time = time(12)
    depth = luigi.FloatParameter()

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    @property
    def metric_id(self):
        return 'UV-%.0fm-CLASS1-BLKS' % self.depth

    def requires(self):
        return ModelData(subset='RFVL', start_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        u = model[input_data().eastward_sea_water_velocity].interpolate(time=time, depth=self.depth)
        v = model[input_data().northward_sea_water_velocity].interpolate(time=time, depth=self.depth)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            u = u(**region)
            v = v(**region)

        mag = np.hypot(u, v)

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = 'Currents at %gm [%s]' % (self.depth,
                                              u.metadata.attributes['units'])

            stride = 6
            plt.contourf(u['longitude'], u['latitude'], mag, 20, cmap='YlOrRd')
            plt.colorbar(orientation='horizontal', label=label)
            plt.quiver(u['longitude'][::stride], u['latitude'][::stride],
                       u[::stride,::stride], v[::stride,::stride],
                       color='k')


class WindStress(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    time = time(12)
    metric_id = 'TAU-CLASS1-BLKS'

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    def requires(self):
        return ModelData(subset='RFVL', start_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        u = model['surface_downward_x_stress'].interpolate(time=time)
        v = model['surface_downward_y_stress'].interpolate(time=time)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            u = u(**region)
            v = v(**region)

        mag = np.hypot(u, v)

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = 'Surface downward wind stress [%s]' % u.metadata.attributes['units']

            stride = 10
            plt.contourf(u['longitude'], u['latitude'], mag, 20, cmap='YlGn_r')
            plt.colorbar(orientation='horizontal', label=label)
            plt.quiver(u['longitude'][::stride], u['latitude'][::stride],
                       u[::stride,::stride], v[::stride,::stride],
                       color='k')


class WindStressCurl(InputUpdateMixin, luigi.Task):
    date = luigi.DateParameter()
    time = time(12)
    metric_id = 'TAUCURL-CLASS1-BLKS'

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - self.date < timedelta(days=30)

    def requires(self):
        return ModelData(subset='RFVL', start_date=self.date)

    def output(self):
        return FigureTarget(self.metric_id, self.date)

    def run(self):
        time = datetime.combine(self.date, self.time)
        model = self.requires().output().open()
        u = model['surface_downward_x_stress'].interpolate(time=time)
        v = model['surface_downward_y_stress'].interpolate(time=time)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            u = u(**region)
            v = v(**region)

        # Mask undefined values since 0 results in large derivatives
        u = np.ma.masked_equal(u, 0.)
        v = np.ma.masked_equal(v, 0.)

        # Calculate derivatives
        dudy = np.diff(u, axis=0) / (np.diff(u['latitude']).reshape(-1, 1) * METER_PER_DEGREE[0])
        dvdx = np.diff(v, axis=1) / (np.diff(u['longitude']).reshape(1, -1) * METER_PER_DEGREE[1])

        # Now moving all variables back on the same grid
        lon = (u['longitude'][1:] + u['longitude'][:-1]) / 2.
        lat = (u['latitude'][1:] + u['latitude'][:-1]) / 2.
        dudy = (dudy[:,:-1] + dudy[:,1:]) / 2.
        dvdx = (dvdx[:-1,:] + dvdx[1:,:]) / 2.

        curl = (dvdx - dudy) * 1e6

        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            label = u'Wind stress curl [10⁻⁶ N/m3]'

            vmax = 15. #np.abs(curl).max()
            levels = np.linspace(-vmax, vmax, 31)
            plt.contourf(lon, lat, curl, levels=levels, cmap='BrBG', extend='both')
            plt.colorbar(orientation='horizontal', label=label)


class ProfileLocations(luigi.Task):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    metric_id = 'S-CLASS1-ARGO-BLKS'

    def requires(self):
        return ProfileObservations(start_date=self.start_date, end_date=self.end_date)

    def run(self):
        with self.output() as fig:
            ax = plt.axes(projection='mercator')
            ax.coastline(visualisation().coastline, sea=None, zorder=2)

            for observation in self.requires().output().iterate('PSAL', [1, 2]):
                if observation['time'] < np.datetime64(self.start_date):
                    continue
                if observation['time'] >= np.datetime64(self.end_date) + np.timedelta64(1, 'D'):
                    continue

                plt.plot(observation['longitude'], observation['latitude'], 'ro', mec='k', mew=0.5)

            plt.title('ARGO profile locations for {:%Y-%m-%d} - {:%Y-%m-%d}'.format(self.start_date, self.end_date))

    def output(self):
        return FigureTarget(self.metric_id, self.start_date, self.end_date)
            

class Hovmoeller(InputUpdateMixin, luigi.Task):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    depth = luigi.FloatParameter(default=10.)
    latitude = luigi.FloatParameter(default=44.)

    @property
    def check_input_update(self):
        # Check input files for changes for up to 30 days
        return datetime.now().date() - (self.end_date or self.start_date) < timedelta(days=30)

    @property
    def metric_id(self):
        return 'S-%.0fm-CLASS4-HOVTEST' % self.depth

    def requires(self):
        return ModelData(subset='PSAL', start_date=self.start_date, end_date=self.end_date)

    def output(self):
        return FigureTarget(self.metric_id, self.start_date, self.end_date)

    def run(self):
        model = self.requires().output().open()
        var = model[input_data().sea_water_salinity].interpolate(depth=self.depth, latitude=self.latitude)
        var = var.moveaxis('time', 0)

        # Selecting only a region of the input data
        region = input_data().region
        if region:
            var = var(**region)

        with self.output() as fig:
            label = '%s at %gm [%s]' % (var.metadata.attributes['long_name'].capitalize(),
                                        self.depth,
                                        var.metadata.attributes['units'].replace('degrees_', u'°'))

            xmask = np.any(var.mask, axis=0)
            x = var['longitude'][~xmask]
            xmin, xmax = x.min(), x.max()

            plt.contourf(var['longitude'], var['time'], var, 10, cmap='PuOr_r')
            plt.colorbar(orientation='vertical', label=label, fraction=0.08, pad=0.04)
            plt.xlim(xmin, xmax)

            plt.gca().xaxis.set_major_formatter(lon_formatter)
            plt.gca().invert_yaxis()


