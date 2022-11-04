#!/usr/bin/env python

from statistics import TimeSeries

ts = TimeSeries()
ts.write('empty.nc', format='NETCDF4_CLASSIC')
