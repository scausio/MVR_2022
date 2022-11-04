# -*- coding: utf-8 -*-
import luigi
from luigi.task import flatten
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from dataset import FancyArray
from .metrics import metrics
from .external import ModelData, input_data
from .targets import OutputDatasetTarget, FigureTarget
from .merger import MergeDataset

try:
    from inflection import underscore
except ImportError:
    from re import split
    def underscore(text):
        groups = filter(lambda g: g, split('([A-Z]+)', text))
        groups[2::2] = ['_%s' % p.lower() for p in groups[2::2]]
        return ''.join(groups).lower()


class BasinScaleDiagnostic(luigi.Task):
    date = luigi.DateParameter()
    metric_id = 'BASIN-DIAG'
    subset = 'TEMP'

    @property
    def resources(self):
        # No concurrent writing to the same file
        return {path: 1 for path in flatten(self.output().path)}

    @property
    def time(self):
        return datetime.combine(self.date, time(12))

    @property
    def name(self):
        return underscore(self.__class__.__name__)
        
    def requires(self):
        return ModelData(subset=self.subset, start_date=self.date, end_date=self.date)

    def output(self):
        return OutputDatasetTarget(self.metric_id, self.date)

    def complete(self):
        try:
            data = self.output().open()
            return self.name in data.variables
        except:
            return False

    def run(self):
        data = self.output().open(mode='r+' if self.output().exists() else 'w')
        data.unlimited = 'time'
        value, units = self.calculate()
        data[self.name] = FancyArray(value, dimensions={'time': np.datetime64(self.time)},
                                            attributes={'units': units})


class VolumeMeanDiagnostic(BasinScaleDiagnostic):

    @property
    def volume(self):
        with nc.Dataset(metrics().mesh_mask) as data:
            e1 = data['e1t'][0]
            e2 = data['e2t'][0]
            e3 = data['e3t'][0]
            mask = data['tmask'][0]

        area = e1 * e2
        volume = area[np.newaxis,...] * e3
        return volume * mask

    def calculate(self):
        volume = self.volume
        variable, units = self.variable
        return (variable * volume).sum() / volume.sum(), units


class AreaMeanDiagnostic(BasinScaleDiagnostic):

    @property
    def area(self):
        with nc.Dataset(metrics().mesh_mask) as data:
            e1 = data['e1t'][0]
            e2 = data['e2t'][0]
            mask = data['tmask'][0,0]

        area = e1 * e2
        return area * mask

    def calculate(self):
        area = self.area
        variable, units = self.variable
        return (variable * area).sum() / area.sum(), units


class VolumeMeanTemperature(VolumeMeanDiagnostic):
    subset = 'TEMP'

    @property
    def variable(self):
        model = self.requires().output().open()
        variable = model[input_data().sea_water_temperature].interpolate(time=self.time)
        return variable, variable.metadata.attributes['units']
    

class VolumeMeanSalinity(VolumeMeanDiagnostic):
    subset = 'PSAL'

    @property
    def variable(self):
        model = self.requires().output().open()
        variable = model[input_data().sea_water_salinity].interpolate(time=self.time)
        return variable, variable.metadata.attributes['units']


class VolumeMeanCurrents(VolumeMeanDiagnostic):
    subset = 'RFVL'

    @property
    def variable(self):
        model = self.requires().output().open()
        u = model[input_data().eastward_sea_water_velocity].interpolate(time=self.time)
        v = model[input_data().northward_sea_water_velocity].interpolate(time=self.time)
        return np.hypot(u, v), u.metadata.attributes['units']


class VolumeMeanKineticEnergy(VolumeMeanDiagnostic):
    subset = 'RFVL'

    @property
    def variable(self):
        model = self.requires().output().open()
        u = model[input_data().eastward_sea_water_velocity].interpolate(time=self.time)
        v = model[input_data().northward_sea_water_velocity].interpolate(time=self.time)

        # Change to cm/s, but only if we are sure the units are m/s
        if u.metadata.attributes['units'] == 'm/s':
            u *= 100.
            v *= 100.
            units = 'cm^2/s^2'
        else:
            units = '(%s)^2' % u.metadata.attributes['units']

        return (u ** 2. + v ** 2.) / 2., units

class BasinMeanSSH(AreaMeanDiagnostic):
    subset = 'ASLV'

    @property
    def variable(self):
        model = self.requires().output().open()
        variable = model[input_data().sea_surface_height].interpolate(time=self.time)
        return variable, variable.metadata.attributes['units']


class BasinMeanUpwardWaterFlux(AreaMeanDiagnostic):
    subset = 'TEMP'

    @property
    def variable(self):
        model = self.requires().output().open()
        variable = model['water_flux_out_of_sea_ice_and_sea_water'].interpolate(time=self.time)
        return variable, variable.metadata.attributes['units']


class BasinMeanDownwardHeatFlux(AreaMeanDiagnostic):
    subset = 'TEMP'

    @property
    def variable(self):
        model = self.requires().output().open()
        variable = model['surface_downward_heat_flux_in_sea_water'].interpolate(time=self.time)
        return variable, variable.metadata.attributes['units']


class BasinMetricTimeSeries(luigi.Task):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    task = luigi.Parameter()

    def requires(self):
        return MergeDataset(**self.param_kwargs)

    @property
    def task_class(self):
        return self.requires().task_class

    @property
    def name(self):
        return self.task_class(date=self.start_date).name

    def output(self):
        parent_id = self.task_class(date=self.start_date).metric_id
        metric_id = '%s_%s' % (parent_id, self.name.upper())
        return FigureTarget(metric_id, self.start_date, self.end_date)

    def run(self):
        data = self.requires().output().open()
        name = self.name
        with self.output() as fig:
            plt.plot(data['time'].astype(datetime),
                     data[name], 'r-')
            label = name.replace('_', ' ').capitalize()
            units = data[name].metadata.attributes['units'].replace('degree_', u'°')
            plt.grid(True, linestyle=':')
            plt.ylabel('%s [%s]' % (label, units))
            fig.autofmt_xdate()


class AllBasinMetricTimeSeries(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()

    def requires(self):
        # Automatically add all tasks derived from AreaMeanDiagnostic, VolumeMeanDiagnostic
        for task in globals().values():
            try:
                if issubclass(task, (AreaMeanDiagnostic, VolumeMeanDiagnostic)):
                    if task not in (AreaMeanDiagnostic, VolumeMeanDiagnostic):
                        yield BasinMetricTimeSeries(task=task.__name__, **self.param_kwargs)
            except TypeError:
                pass

