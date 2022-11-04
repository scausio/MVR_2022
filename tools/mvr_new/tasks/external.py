import luigi
import numpy as np
import dataset
from glob import glob
from datetime import date, timedelta
from luigi.task import flatten
from re import split
from .targets import *
from .util import OrderedSet, date_range


class input_data(luigi.Config):
    satellite_sst = luigi.Parameter()
    satellite_alt = luigi.Parameter()
    insitu = luigi.Parameter()
    reprocessing_delay = luigi.IntParameter(default=14)
    altimetry_satellites = luigi.Parameter(default='AltiKa, Cryosat-2, Jason-2, Jason-3, Sentinel-3A')
    
    model = luigi.Parameter()
    subset_amxl = luigi.Parameter(default='AMXL')
    subset_aslv = luigi.Parameter(default='ASLV')
    subset_psal = luigi.Parameter(default='PSAL')
    subset_rfvl = luigi.Parameter(default='RFVL')
    subset_temp = luigi.Parameter(default='TEMP')

    sea_water_temperature = luigi.Parameter(default='sea_water_potential_temperature')
    sea_water_salinity = luigi.Parameter(default='sea_water_salinity')
    eastward_sea_water_velocity = luigi.Parameter(default='eastward_sea_water_velocity')
    northward_sea_water_velocity = luigi.Parameter(default='northward_sea_water_velocity')
    sea_surface_height = luigi.Parameter(default='sea_surface_height_above_sea_level')
    mixed_layer_depth = luigi.Parameter(default='ocean_mixed_layer_thickness')

    region_longitude = luigi.Parameter(default=None)
    region_latitude = luigi.Parameter(default=None)
    region_depth = luigi.Parameter(default=None)

    interpolation_method = luigi.Parameter(default='linear')
    ignore_missing = luigi.BoolParameter(default=False)

    def _region_select(self, name):
        param = getattr(self, 'region_%s' % name)
        if param is None:
            return None
        try:
            val_min, val_max = map(float, split('[ ,;]+', param))
            return slice(val_min, val_max)
        except Exception as e:
            raise RuntimeError('region_X should contain a comma separated min and max coordinate')

    @property
    def region(self):
        select = {}
        for dimname in ['longitude', 'latitude', 'depth']:
            dim_select = self._region_select(dimname)
            if dim_select is not None:
                select[dimname] = dim_select

        return select

    @property
    def altimetry_satellites_list(self):
        return split('[ ,;]+', self.altimetry_satellites)


dataset.INTERPOLATION_METHOD = input_data().interpolation_method


class ProfileObservations(luigi.ExternalTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=None)

    def complete(self):
        return True

    def output(self):
        #TODO: handle the case where there is no data available
        dates = date_range(self.start_date, self.end_date or self.start_date)
        files = OrderedSet(input_data().insitu.format(date=date) for date in dates)
        return InSituDatasetTarget(files)


class SatelliteSSTObservation(luigi.ExternalTask):
    date = luigi.DateParameter()

    def complete(self):
        if input_data().ignore_missing:
            return True
        else:
            return super(SatelliteSSTObservation, self).complete()

    def output(self):
        return DatasetTarget(input_data().satellite_sst.format(date=self.date))


class SatelliteSLAObservation(luigi.ExternalTask):
    date = luigi.DateParameter()

    def _find_files(self, satellite):
        production_start_date = self.date
        production_end_date = self.date + timedelta(days=input_data().reprocessing_delay)
        for production_date in list(date_range(production_start_date, production_end_date))[::-1]:
            files = glob(input_data().satellite_alt.format(date=self.date, satellite=satellite, production_date=production_date))
            if len(files) > 0:
                return files

    def complete(self):
        if input_data().ignore_missing:
            return True
        else:
            return super(SatelliteSLAObservation, self).complete()

    def output(self):
        files = flatten([self._find_files(sat) for sat in input_data().altimetry_satellites_list])
        return SatelliteDatasetTarget(files)


class ModelData(luigi.ExternalTask):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter(default=None)
    subset = luigi.Parameter(default='TEMP')
    add_extra_days_before = luigi.IntParameter(default=0)
    add_extra_days_after = luigi.IntParameter(default=0)
    fix_coordinates = luigi.BoolParameter(default=False)
    time_offset = luigi.IntParameter(default=0)
    calendar = luigi.Parameter(default='standard')

    def output(self):
        try:
            subset = getattr(input_data(), 'subset_%s' % self.subset.lower())
        except AttributeError:
            raise ValueError('no such subset "%s"' % self.subset)

        dates = date_range(self.start_date - timedelta(days=self.add_extra_days_before), 
                           (self.end_date or (self.start_date + timedelta(days=1))) + timedelta(days=self.add_extra_days_after))
        files = OrderedSet(input_data().model.format(date=date, subset=subset) for date in dates)
        return DatasetTarget(files, fix_coordinates=self.fix_coordinates, time_offset=self.time_offset, calendar=self.calendar)

