from scipy.stats import binned_statistic
from dataset import *
from datetime import datetime, date
import matplotlib.pyplot as plt
import numpy as np


class Statistic(object):
    """
    Aggregates data into statistics such as RMSD, BIAS etc.
    """

    ATTRIBUTES = ('bias', 'rmsd', 'count')

    def __init__(self, bins=None, **kwargs):
        if bins is None:
            bins = [-1e20, 1e20]
        self.bins = np.asanyarray(bins, dtype=float)

        for attr in self.ATTRIBUTES:
            value = kwargs.pop(attr, np.array(0.)).view(np.ndarray)
            setattr(self, attr, value)

    def __iadd__(self, other):
        if not np.all(self.bins == other.bins):
            raise ValueError('cannot add statistics with different bins')

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.ma.masked_invalid(self.count / (self.count + other.count)).filled(0.)
        self.bias = ratio * self.bias + (1. - ratio) * other.bias
        self.rmsd = np.sqrt(ratio * self.rmsd ** 2. + (1. - ratio) * other.rmsd ** 2.)
        self.count = self.count + other.count

        return self

    def __repr__(self):
        return 'Statistic(bias={!r}, rmsd={!r}, count={!r}) object at 0x{:x}'.format(self.bias,
                                                                                     self.rmsd,
                                                                                     self.count,
                                                                                     id(self))

    def __add__(self, other):
        # Implement __add__ using __iadd__
        result = self.copy()
        result += other
        return result

    def add(self, misfit, depth, unbiased=False):
        if misfit.shape != depth.shape:
            raise ValueError('misfit and depth should have the same size')

        if not np.ma.isMaskedArray(misfit):
            misfit = np.ma.masked_invalid(misfit)

        count = binned_statistic(depth, ~misfit.mask, statistic='sum', bins=self.bins)[0]
        if unbiased:
            # When the mean of the observations is subtracted prior to
            # comparing, the resulting RMSD is underestimated. The unbiased
            # version of the RMSD takes this into account by dividing by N-1
            # instead of N.
            count[count > 0] -= 1
        safe_count = np.ma.masked_equal(count, 0.).filled(1.)

        kwargs = {'bias': binned_statistic(depth, misfit.filled(0.), statistic='sum', bins=self.bins)[0] / safe_count,
                  'rmsd': np.sqrt(binned_statistic(depth, misfit.filled(0.) ** 2., statistic='sum', bins=self.bins)[0] / safe_count),
                  'count': count}

        self += Statistic(bins=self.bins, **kwargs)

    def copy(self):
        return Statistic(bins=self.bins, **{attr: np.copy(getattr(self, attr)) for attr in self.ATTRIBUTES})

    @property
    def rmse(self):
        return self.rmsd


class TimeSeries(object):
    # contains a series of metrics as a function of time

    def __init__(self, dataset=None, bins=None, units='datetime64[D]', **kwargs):
        self.units = units
        self.data = {}

        if dataset is not None:
            if not isinstance(dataset, Dataset):
                dataset = Dataset(dataset)

            bins = dataset['depth_bnds']
            # FIXME: np.concatenate breaks FancyArray
            self.bins = np.concatenate([bins(bounds=0), [bins(bounds=1, depth=-1)]]).view(np.ndarray)

            for index, time in enumerate(dataset['time']):
                stat = Statistic(bins=self.bins, **{attr: dataset[attr](time=time) for attr in Statistic.ATTRIBUTES})
                self[time] = stat

        else:
            if bins is None:
                bins = [-1e20, 1e20]
            self.bins = np.asanyarray(bins)

    def change_units(self, units):
        timeseries = TimeSeries(bins=self.bins, units=units)
        for time, value in self.data.iteritems():
            # FIXME: if you do timeseries[time] += value also TimeSeries.__setitem__ will be called (??!)
            stat = timeseries[time]
            stat += value

        return timeseries

    def __getitem__(self, time):
        if isinstance(time, date):
            time = np.datetime64(time)

        time = time.astype(self.units)

        try:
            item = self.data[time]
        except KeyError:
            item = Statistic(bins=self.bins)
            self.data[time] = item

        return item

    def __setitem__(self, time, item):
        if isinstance(time, date):
            time = np.datetime64(time)

        time = time.astype(self.units)

        try:
            self.data[time] += item
        except KeyError:
            self.data[time] = item.copy()

    def __iter__(self):
        for date in sorted(self.data.keys()):
            yield date, self.data[date]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'Timeseries(%s)' % ', '.join(map(str, self.data.keys()))

    def write(self, output, **kwargs):
        if isinstance(output, Dataset):
            dataset = output
        else:
            dataset = Dataset(output, mode='w', **kwargs)

        # TODO: refactor so that we don't set this to a dimension that doesn't exist yet
        dataset.unlimited = 'time'

        times = np.array(sorted(self.data.keys()), dtype='datetime64')
        depth = (self.bins[1:] + self.bins[:-1]) / 2.

        depth_dims = OrderedDict((('depth', depth), ('bounds', np.arange(2.))))
        depth_bnds = FancyArray(np.vstack([self.bins[:-1], self.bins[1:]]).T,
                                dimensions=depth_dims)
        dataset['depth_bnds'] = depth_bnds

        dims = OrderedDict((('time', times), ('depth', depth)))
        for attr in Statistic.ATTRIBUTES:
            if len(times) > 0:
                values = np.vstack([getattr(self.data[time], attr) for time in times])
            else:
                values = np.zeros((0, depth.size))
            array = FancyArray(values, dimensions=dims)
            dataset[attr] = array


    def get_metrics(self, *metrics):
        if not self.data:
            return (None,) * (len(metrics)+1)

        dates = np.array(sorted(self.data.keys())) 
        data = tuple(np.vstack([getattr(self.data[date], metric) for date in dates]) for metric in metrics)
        return (dates,) + data


import unittest

class StatisticTest(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(10000.)
        self.y = np.random.uniform(self.x.min(), self.x.max(), size=self.x.shape)
        self.stat = Statistic()
        self.stat.add(self.x - self.y, np.zeros_like(self.x))

    def testRMSE(self):
        rmse = np.sqrt(np.power(self.x - self.y, 2.).mean())
        np.testing.assert_almost_equal(self.stat.rmse[0], rmse)

    def testBias(self):
        bias = (self.x - self.y).mean()
        np.testing.assert_almost_equal(self.stat.bias[0], bias)


class TimeSeriesTest(unittest.TestCase):
    def setUp(self):
        x = np.arange(100.)
        y = np.random.uniform(x.min(), x.max(), size=x.shape)
        z = np.random.normal(x.mean(), x.std(), size=x.shape)

        self.stat1 = Statistic()
        self.stat1.add(x - y, np.zeros_like(x))
        self.stat2 = Statistic()
        self.stat2.add(y - z, np.zeros_like(x))

        self.timeseries = TimeSeries()
        self.timeseries[np.datetime64('2018-01-01')] = self.stat1
        self.timeseries[np.datetime64('2018-01-02')] = self.stat2

    def testChangeUnits(self):
        newunits = self.timeseries.change_units('datetime64[W]')
        self.assertEquals(newunits.data.values()[0].count.squeeze(), 200)

    def testWrite(self):
        self.timeseries.write('timeseries_test.nc')

if __name__ == '__main__':
    unittest.main()
