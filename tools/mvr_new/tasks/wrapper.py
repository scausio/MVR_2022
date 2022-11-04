# -*- coding: utf-8 -*-
import luigi
import numpy as np
from datetime import datetime, date, timedelta

from tools.mvr_new.tasks.metrics import *
from tools.mvr_new.tasks.visualisation import *
from tools.mvr_new.tasks.basinscale import *
from tools.mvr_new.tasks.util import date_range


class UpdateCalval(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    frequency = luigi.ChoiceParameter(default='D', choices=['D', 'W', 'M', 'Y'])

    @property
    def end_date(self):
        day = datetime.now().date()

        # Analysis goes until the Monday of the week before...
        delta = timedelta(days=day.weekday() + 7)

        # ...and is ready only after Wednesday
        if day.weekday() < 2:
            delta += timedelta(days=7)

        return day - delta

    def requires(self):
        yield AllMetrics(start_date=self.start_date,
                         end_date=self.end_date,
                         frequency=self.frequency)


class AllTasks(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=datetime.now().date())

    def requires(self):
        for frequency in ['D', 'W', 'M', 'Y']:
            yield AllMetrics(start_date=self.start_date, end_date=self.end_date, frequency=frequency)
        yield AllVisualisation(start_date=self.start_date, end_date=self.end_date)


class AllMetrics(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=datetime.now().date() - timedelta(days=1))
    frequency = luigi.ChoiceParameter(default='D', choices=['D', 'W', 'M', 'Y'])

    def requires(self):
        params = {'start_date': self.start_date,
                  'end_date': self.end_date,
                  'frequency': self.frequency}

        yield SatelliteSSTMetrics(**params)
        yield SatelliteSLAMetrics(**params)
        yield AllProfileMetrics(**params)


class AllProfileMetrics(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=datetime.now().date() - timedelta(days=1))
    frequency = luigi.ChoiceParameter(default='D', choices=['D', 'W', 'M', 'Y'])

    def requires(self):
        params = {'start_date': self.start_date,
                  'end_date': self.end_date,
                  'frequency': self.frequency}

        for layer in range(len(metrics().layers_array) - 1):
            params['layer'] = layer
            yield TemperatureProfileMetrics(**params)
            yield SalinityProfileMetrics(**params)

        yield ProfileLocations(start_date=self.start_date, end_date=self.end_date)


class Diagnostics(luigi.WrapperTask):
    date = luigi.DateParameter()

    def requires(self):
        yield VolumeMeanTemperature(date=self.date)
        yield VolumeMeanSalinity(date=self.date)
        yield VolumeMeanCurrents(date=self.date)
        yield VolumeMeanKineticEnergy(date=self.date)
        yield BasinMeanSSH(date=self.date)
        yield BasinMeanUpwardWaterFlux(date=self.date)
        yield BasinMeanDownwardHeatFlux(date=self.date)


class AllVisualisation(luigi.WrapperTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=None)

    def requires(self):
        for date in date_range(self.start_date, self.end_date):
            yield Visualisation(date=date)


class Visualisation(luigi.WrapperTask):
    date = luigi.DateParameter()
    extra_diagnostics = luigi.BoolParameter(default=False)

    def requires(self):
        yield MixedLayerDepth(date=self.date)
        yield SeaSurfaceHeight(date=self.date)
        yield Transects(date=self.date)
        if self.extra_diagnostics:
            yield UpwardWaterFlux(date=self.date)
            yield ShortwaveRadiation(date=self.date)
            yield DownwardHeatFlux(date=self.date)
            yield WindStress(date=self.date)
            yield WindStressCurl(date=self.date)
            yield Diagnostics(date=self.date)
        for depth in visualisation().depth_levels_array:
            yield Temperature(date=self.date, depth=depth)
            yield Salinity(date=self.date, depth=depth)
            yield Currents(date=self.date, depth=depth)

