"""NetCDF access with named dimensions and automatic interpolation."""

__author__ = "Eric Jansen"
__email__ = "eric.jansen@cmcc.it"

import netCDF4 as nc
import numpy as np
import functools
import warnings
from itertools import chain
from collections import OrderedDict
from scipy.interpolate import interp1d


DATETYPE = 'datetime64[s]'
DATEUNITS = 'seconds since 1970-01-01 00:00'
INTERPOLATION_METHOD = 'linear'


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


class InterpolationError(Exception):
    pass


def nearest(a, b):
    """
    Find value in a closest to b and return its index.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    if a.dtype != b.dtype:
        # FIXME: unsure why this is needed, seems datetime64[s] cannot be converted into itself(?)
        b = a.dtype.type(b)
    return np.argmin(np.abs(b - a))


def regrid(*grids, **kwargs):
    """
    Regrid arrays in preparation for addition, subtraction etc.

    Perform axis permutations and interpolation to match the coordinates and
    axis order of the input arrays. Axis order will be taken from the first
    array. Both variables may be modified. Views are returned whenever
    possible.

    Parameters
    ----------
    grids: list of N Lazy/FancyArray (N>1)
        Arrays to regrid.
    dimensions: list of str, optional
        Dimensions to regrid (default: all that are in common).
    how: int or ['best', 'worst']
        Take the resulting dimension from the specified input grid, or use
        the best ('best') or worst ('worst') resolution among all grids.
    """
    dimensions = kwargs.pop('dimensions', tuple(common_dimensions(*grids)))

    interpolation = {}
    for name in dimensions:
        dims = (array.metadata.get_dimension(name) for array in grids)
        interpolation[name] = regrid_dimension(*dims, **kwargs)

    interpolate = lambda a: a.interpolate(**interpolation)
    result = map(interpolate, grids)

    for index, name in enumerate(dimensions):
        moveaxis = lambda a: a.moveaxis(name, index)
        result = map(moveaxis, result)

    return result


def regrid_dimension(*dimensions, **kwargs):
    """
    Decides the best common dimension for a set.
    
    Parameters
    ----------
    dimensions: list of array_like
        Coordinate values for dimension in different grids.
    how: int or ['best', 'worst']
        Take the resulting dimension from the specified input grid, or use
        the best ('best') or worst ('worst') resolution among all grids.
    """
    how = kwargs.pop('how', 'worst')
    rmin = np.max([dim.min() for dim in dimensions])
    rmax = np.min([dim.max() for dim in dimensions])

    if how in ('best', 'worst'):
        # TODO: It could be nice to focus mean_distance on the correct range
        dist = [np.mean(mean_distance(dim)) for dim in dimensions]

        if how == 'worst':
            how = np.argmax(dist)
        else:
            how = np.argmin(dist)

    result = np.copy(dimensions[how])

    # Do not perform this for a single value, as it will turn it into an array
    if result.shape:
        result = result[np.logical_and(result >= rmin, result <= rmax)]
    
    return result


def common_dimensions(*grids):
    """
    Yields the names of the dimensions that grids have in common.

    Resulting names may be names, aliases or a mix of both.
    """
    if len(grids) < 2:
        raise ValueError('two or more grids are required')

    for dim in grids[0].metadata.dimensions.keys():
        # Possible names are the true name or any alias that points to it
        names = chain((dim,), (alias for alias, name in grids[0].metadata.aliases.iteritems() if name == dim)) 

        for name in names:
            try:
                for array in grids[1:]:
                    array[name]
            except KeyError:
                continue

            yield name
            break


def mean_distance(a):
    """
    Calculate the mean distance to the points left and right for each point on
    the axis a.
    """
    dist = np.zeros(shape=a.shape, dtype='f8')

    if not hasattr(a, '__len__') or not a.shape:
        return np.inf
    elif len(a) > 1:
        diff = (a[1:] - a[:-1]).astype('f8')
        dist[1:] += diff
        dist[:-1] += diff
        dist[1:-1] /= 2.
    else:
        dist[:] = np.inf

    return dist


class PrintOptions(object):
    def __init__(self, **opts):
        self.options = opts

    def __enter__(self):
        self.default = np.get_printoptions()
        np.set_printoptions(**self.options)

    def __exit__(self, *args, **kwargs):
        np.set_printoptions(**self.default)


class MetaData(object):
    def __init__(self, dimensions=None, attributes=None, aliases=None, unstructured=False):
        self.dimensions = dimensions or OrderedDict()
        self.attributes = attributes or OrderedDict()
        self.aliases = aliases or dict()
        self.unstructured = unstructured

        #TODO: move sanity checks to a separate method so we can check also when changes are made
        if unstructured and len(set(dim.shape for dim in self.dimensions.values() if dim.shape)) > 1:
            raise ValueError('unstructured arrays require all dimensions to have the shape of the data')

    @property
    def scalar_mask(self):
        return np.array([not dim.shape for dim in self.dimensions.values()])

    @property
    def shape(self):
        if self.unstructured:
            for dim in self.dimensions.values():
                if dim.shape:
                    return dim.shape
            else:
                raise RuntimeError('this is not supposed to happen')
        else:
            return tuple(dim.size for dim in self.dimensions.values() if dim.shape)

    def copy(self):
        return MetaData(dimensions=self.dimensions.copy(),
                        attributes=self.attributes.copy(),
                        aliases=self.aliases.copy(),
                        unstructured=self.unstructured)

    def __repr__(self):
        return 'MetaData({!r}) object at 0x{:x}'.format(self.dimensions.keys(), id(self))

    def resolve_alias(self, name):
        if not name in self.dimensions:
            name = self.aliases.get(name, name)
        return name

    def get_dimension(self, name):
        if isinstance(name, (int, long)):
            return self.dimensions.values()[name]
        else:
            name = self.resolve_alias(name)
            return self.dimensions[name]

    def set_dimension(self, name, values):
        name = self.resolve_alias(name)
        self.dimensions[name] = np.asarray(values)

    def get_index(self, name):
        name = self.resolve_alias(name)
        return self.dimensions.keys().index(name)

    def is_scalar(self, name):
        dim = self.get_dimension(name)
        return not dim.shape

    def axis_from_index(self, index):
        mask = self.scalar_mask
        # Axis is the index, minus the number of scalar dimensions before it
        #print self.dimensions, '-> axis_from_index(',index,') =', index - mask[:index].sum()
        return index - mask[:index].sum()

    def index_from_axis(self, axis):
        mask = self.scalar_mask
        return np.argmax(mask.cumsum() == axis)

    # FIXME: rename select -> item
    def __getitem__(self, select):
        if not isinstance(select, (tuple, list)):
            select = (select,)

        meta = MetaData(aliases=self.aliases, attributes=self.attributes, unstructured=self.unstructured)

        if self.unstructured:
            select *= (self.scalar_mask == False).sum()

        sel = iter(select)
        for name, dim in self.dimensions.iteritems():
            if dim.shape:
                try:
                    dimsel = sel.next()
                    if isinstance(dimsel, tuple):
                        # FIXME: here it should become unstructured
                        dimsel = dimsel,
                    
                    dim = dim[dimsel]
                except StopIteration:
                    pass
            meta.dimensions[name] = dim

        return meta

    def nearest(self, select):
        select = self._expand_all(select)

        index = tuple(select.pop(name, slice(None)) for name, dim
                                                    in self.dimensions.iteritems()
                                                    if dim.shape)

        if len(select):
            raise IndexError('dimensions not found: {}'.format(select.keys()))

        return index

    def _expand_all(self, select):
        expanded = {}
        for name, sel in select.iteritems():
            if not name in self.dimensions.keys():
                name = self.aliases[name]
            expand = self._expand(name, sel)

            if expand is not None:
                expanded[name] = expand

        return expanded

    def _expand(self, name, select):
        if isinstance(select, (int, long)) or select == Ellipsis:
            return select

        dim = self.dimensions[name]
        
        if isinstance(select, slice):
            start = nearest(dim, select.start) if not isinstance(select.start, (int, long)) else select.start
            stop = nearest(dim, select.stop) + 1 if not isinstance(select.stop, (int, long)) else select.stop
            return slice(start, stop, select.step)
        else:
            return nearest(dim, select)


class NamedDimensionsMixin(object):
    """Data access methods using named dimensions."""

    def __call__(self, **select):
        """__getitem__ with named dimensions and indices"""
        return self.nearest(**select)

    def nearest(self, **select):
        """__getitem__ with named dimensions and values"""
        index = self.metadata.nearest(select)
        return self[index]

    def interpolate(self, **dimensions):
        """__getitem__ with named dimensions and interpolation"""

        # First we roughly cut the domain to size for optimal performance, if
        # the data is in a (remote) netCDF4 file only the area around the
        # target location needs to be loaded in memory
        select = {}
        for name, values in dimensions.iteritems():
            dim = self.metadata.get_dimension(name)

            # For np.datetime64 values may be str, convert explicitly
            values = np.asanyarray(values, dtype=dim.dtype)

            if values.size == 0:
                raise InterpolationError('no valid points remain after interpolation')

            if dim.shape:
                imin = max(np.searchsorted(dim, values.min(), side='left') - 1, 0)
                imax = min(np.searchsorted(dim, values.max(), side='right') + 1, len(dim))
                select[name] = slice(imin, imax)

        result = self(**select)

        # Iterate over the axes and perform interpolation
        for name, values in dimensions.iteritems():
            dim = self.metadata.get_dimension(name)
            if np.all(np.asanyarray(values) == dim):
                # FIXME: should be done in a more general way for better efficiency
                if np.ndim(values) == 0 and np.ndim(dim) != 0:
                    result = result(**{name: 0})
                continue
            #print 'Going to interpolate %s from %s to %s' % (name, dim, values)
            result = result.interpolate_dimension(name, values)

        return result

    def diagonal_points(self):
        """
        Take diagonal elements from an array to convert a structured into an unstructured array.
        """
        if not np.all(np.diff(self.shape) == 0):
            raise ValueError('size of all dimensions should equal the number of dimensions')

        size = self.shape[0]
        index = (range(size),) * self.ndim
        result = self[index]
        result.metadata.unstructured = True

        return result



class LazyArray(NamedDimensionsMixin):
    """Array wrapper that postpones data reading as much as possible."""

    # This maskes sure that np.ma.asanyarray does not wrap this class again
    _baseclass = np.ndarray

    def __init__(self, data, **kwargs):
        dimensions = kwargs.pop('dimensions', None)
        attributes = kwargs.pop('attributes', None)
        self.metadata = kwargs.pop('metadata', MetaData(dimensions=dimensions,
                                                        attributes=attributes))
        self.data = data

    def __array__(self, **kwargs):
        return FancyArray(self.data[...], metadata=self.metadata, **kwargs)

    def __getitem__(self, item):
        # String labels mean retrieving dimension variables
        if isinstance(item, (str, unicode)):
            return self.metadata.get_dimension(item)

        # TODO: LazyArray can carry a select attribute to store one selection
        #       per dimension, this way it can remain lazy somewhat longer
        result = self.data[item].view(FancyArray)
        if isinstance(result, FancyArray):
            result.metadata = self.metadata[item]
        return result

    def __getattr__(self, attr):
        if attr in ('__eq__', '__ne__', 'filled'):
            # For any of the mathematical operations, load the data
            return getattr(np.asanyarray(self), attr)

        if not attr.startswith('_'):
            try:
                #print '-> passing on call to %s' % attr
                return getattr(self.data, attr)
            except AttributeError:
                pass

        raise AttributeError(attr)

    def __sub__(self, other):
        # FIXME: check how this should work with copy/view
        if isinstance(other, (FancyArray, LazyArray)):
            result, other = regrid(self, other)
        result = result.copy()
        result -= np.asanyarray(other)
        return result

    def __add__(self, other):
        # FIXME: check how this should work with copy/view
        if isinstance(other, (FancyArray, LazyArray)):
            result, other = regrid(self, other)
        result = result.copy()
        result += np.asanyarray(other)
        return result

    def __str__(self):
        with PrintOptions(threshold=10, linewidth=120):
            output = ['Values:']
            output.append('    ... (%d values, not loaded yet) ...' % self.size)
            output.append('Attributes:')
            for name, attr in self.metadata.attributes.iteritems():
                output.append('    %-16s %s' % (name, attr))
            output.append('Coordinates%s:' % (' (unstructured)' if self.metadata.unstructured else ''))
            for name, dim in self.metadata.dimensions.iteritems():
                output.append('    %-16s (%s) %s (%d)' % (name, dim.dtype, dim, dim.size))

        return '\n'.join(output)
    __repr__ = __str__

    def copy(self, **kwargs):
        return np.asanyarray(self).copy()

    @property
    def size(self):
        return np.prod(self.shape)


def _wrap_function(func):
    def method(self, **kwargs):
        try:
            dim = kwargs.pop('dim')
            if isinstance(dim, str):
                dim = (dim,)
            
            kwargs['axis'] = tuple(self.metadata.get_index(d) for d in dim)
        except KeyError:
            pass

        result = func(self, **kwargs)

        if isinstance(result, FancyArray):
            result.metadata = self.metadata.copy()
            for d in dim:
                del result.metadata.dimensions[d]

        return result
    return method


class FancyArray(np.ma.MaskedArray, NamedDimensionsMixin):

    def __new__(cls, *args, **kwargs):
        dimensions = kwargs.pop('dimensions', None)
        attributes = kwargs.pop('attributes', None)
        unstructured = kwargs.pop('unstructured', False)
        metadata = kwargs.pop('metadata', MetaData(dimensions=dimensions,
                                                   attributes=attributes,
                                                   unstructured=unstructured))
        data = np.ma.array(*args, **kwargs).view(cls)

        if not unstructured and data.shape != metadata.shape:
            raise ValueError('dimension shape does not match data shape (%r != %r)' % (metadata.shape, data.shape))

        # some checking on dimensions

        data.metadata = metadata
        return data

    def __array_finalize__(self, obj):
        super(FancyArray, self).__array_finalize__(obj)

        if obj is not None:
            self.metadata = getattr(obj, 'metadata', MetaData())

    def __getitem__(self, item):
        # String labels mean retrieving dimension variables
        if isinstance(item, (str, unicode)):
            return self.metadata.get_dimension(item)

        result = super(FancyArray, self).__getitem__(item)
        if isinstance(result, FancyArray):
            if not hasattr(self, 'metadata'):
                warnings.warn('FancyArray without metadata, where does it come from?')
            else:
                result.metadata = self.metadata[item]
        return result

    def __setitem__(self, item, value):
        # String labels mean setting dimension variables
        if isinstance(item, (str, unicode)):
            return self.metadata.set_dimension(item, value)
        else:
            return super(FancyArray, self).__setitem__(item, value)

    def __str__(self):
        # TODO: this may be merged with the LazyArray __str__
        with PrintOptions(threshold=10, linewidth=120):
            output = ['Values:']
            output.append('    %s' % super(FancyArray, self).__str__().replace('\n', '\n    '))
            output.append('Attributes:')
            for name, attr in self.metadata.attributes.iteritems():
                output.append('    %-16s %s' % (name, attr))
            output.append('Coordinates%s:' % (' (unstructured)' if self.metadata.unstructured else ''))
            for name, dim in self.metadata.dimensions.iteritems():
                output.append('    %-16s (%s) %s (%d)' % (name, dim.dtype, dim, dim.size))

        return '\n'.join(output)
    __repr__ = __str__

    def __sub__(self, other):
        result = self.copy()
        if isinstance(other, (FancyArray, LazyArray)):
            result, other = regrid(result, other)
        result -= np.asanyarray(other)
        return result

    def __add__(self, other):
        result = self.copy()
        if isinstance(other, (FancyArray, LazyArray)):
            result, other = regrid(result, other)
        result += np.asanyarray(other)
        return result

    mean = _wrap_function(np.ma.MaskedArray.mean)
    min = _wrap_function(np.ma.MaskedArray.min)
    max = _wrap_function(np.ma.MaskedArray.max)

    def filled(self, *args, **kwargs):
        fancy = kwargs.pop('fancy', False)
        result = super(FancyArray, self).filled(*args, **kwargs)

        if fancy:
            result = result.view(type(self))
            result.metadata = self.metadata.copy()
        return result

    def copy(self, **kwargs):
        # This is necessary when subclassing np.ma.MaskedArray
        copy = super(FancyArray, self).copy(**kwargs)
        if isinstance(copy, FancyArray):
            copy.metadata = self.metadata.copy()

        return copy

    def interpolate_dimension(self, name, values):
        values = np.asanyarray(values)
        index = self.metadata.get_index(name)
        axis = self.metadata.axis_from_index(index)
        dim = self.metadata.get_dimension(name)

        #print 'Interpolating %s from %r to %r' % (name, dim, values)

        if not dim.shape:
            if not values.shape:
                if dim != values:
                    warnings.warn('value %s != %s for dimension "%s"' % (dim, values, name))
                    pass
                return self
            else:
                raise InterpolationError('cannot interpolate "%s" from %s to %s' % (name, dim, values))

        is_time = np.issubdtype(dim.dtype, np.datetime64)
        if is_time:
            dim = dim.astype(DATETYPE).astype(float)
            values = values.astype(DATETYPE).astype(float)

        # Drop masked values
        if np.ma.isMaskedArray(values):
            values = values.compressed()

        # Drop values outside the current range (no extrapolation)
        valid = np.logical_and(values >= np.min(dim),
                               values <= np.max(dim))
        if valid.shape:
            values = values[valid]
        elif valid == False:
            # No valid values remain, raise an error
            raise InterpolationError('cannot interpolate "%s" from %s to %s' % (name, dim, values))

        # Using self.filled(0) is important, interp1d behaves badly with masked/nan values
        #interp = interp1d(dim, self.filled(0), axis=axis, kind=INTERPOLATION_METHOD)

        # Alternatively replace only the nan values, we might want to unmask values outside valid_min/max
        filled = self.copy()
        filled.mask = False
        filled[filled == np.nan] = 0

        # Replace also _FillValue and missing_value, 1e+20 appears to cause problems sometimes too
        filled[filled == filled.metadata.attributes.get('_FillValue', np.nan)] = 0
        filled[filled == filled.metadata.attributes.get('missing_value', np.nan)] = 0

        interp = interp1d(dim, filled, axis=axis, kind=INTERPOLATION_METHOD)
        data = interp(values)

        # If any masked values were present, they will be masked again here
        mask = False
        if np.ma.is_masked(self):
            # Repeat the process for interpolating the mask
            interp_mask = interp1d(dim, self.mask, axis=axis, kind=INTERPOLATION_METHOD)
            mask = interp_mask(values) > 0

        metadata = self.metadata.copy()
        metadata.set_dimension(name, values.astype(DATETYPE) if is_time else values)

        return FancyArray(data, mask=mask, metadata=metadata)

    def moveaxis(self, source, destination):
        if isinstance(source, (str, unicode)):
            source = self.metadata.get_index(source)

        if source == destination:
            return self

        names = self.metadata.dimensions.keys()
        coord = names.pop(source)
        names.insert(destination, coord)

        if not self.metadata.is_scalar(source):
            #print 'Moving axis %r (%r) to %r (%r)' % (source, self.metadata.axis_from_index(source), destination, self.metadata.axis_from_index(destination))

            result = np.moveaxis(self, 
                                 self.metadata.axis_from_index(source),
                                 self.metadata.axis_from_index(destination))
        else:
            result = self.copy()  # FIXME: is a copy necessary or can the data just be a view?

        dimensions = OrderedDict(((name, self.metadata.dimensions[name]) for name in names))
        result.metadata.dimensions = dimensions
        result.metadata.aliases = self.metadata.aliases
        result.metadata.attributes = self.metadata.attributes

        return result


class PrintOptions(object):
    def __init__(self, **opts):
        self.options = opts

    def __enter__(self):
        self.default = np.get_printoptions()
        np.set_printoptions(**self.options)

    def __exit__(self, *args, **kwargs):
        np.set_printoptions(**self.default)


class Dataset(object):
    def __init__(self, filename, **kwargs):

        self.unlimited = None
        self.calendar = kwargs.pop('calendar', 'standard')

        if isinstance(filename, (list, tuple)) and len(filename) == 1:
            filename = filename[0]

        # Switch between netCDF4.Dataset and netCDF4.MFDataset based on filename
        if isinstance(filename, str) and (not any(c in filename for c in '*?') or filename.startswith('http://')):
            # Single files without wildcards or remote datasets are Dataset
            self.netcdf = nc.Dataset(filename, **kwargs)
        else:
            # All else is MFDataset
            self.netcdf = nc.MFDataset(filename, **kwargs)

        # Read variables that are dimensions (e.g. longitude, latitude)
        self.dimensions = OrderedDict() 
        for name, dim in self.netcdf.dimensions.iteritems():
            if dim.isunlimited():
                self.unlimited = name

            try:
                var = self.netcdf.variables[name]
            except KeyError:
                #warnings.warn('no variable matching dimension "%s", making simple arange' % name)
                self.dimensions[name] = np.arange(len(dim))
                continue

            # Convert time variable into numpy.datetime64
            if (name == 'time' or getattr(var, 'standard_name', None) == 'time') and len(var) > 0:

                # netCDF4.MFTime ensures the units are consistent across multiple files
                if dim.isunlimited() and isinstance(self.netcdf, nc.MFDataset):
                    var = nc.MFTime(var, calendar=self.calendar)
                
                try:
                    units = getattr(var, 'units')
                    calendar = getattr(var, 'calendar', self.calendar)
                    data = nc.num2date(var[...], units, calendar).astype(DATETYPE)
                except (ValueError,IOError) as err:
                    warnings.warn('unable to decode time variable "%s"' % name)

                    # Use variable as it is
                    data = var[...]

            else:
                # Load the contents of the variable (only for dimensions)
                if len(var) > 0:  # Use len(), attribute size does not exist for MFDataset Variables
                    data = var[...]
                else:
                    data = np.empty(shape=var.shape, dtype=var.dtype)

            self.dimensions[name] = data

        # Read the rest of the variables (skip the dimensions loaded before)
        self.aliases = {}
        self.variables = OrderedDict()
        for name, var in self.netcdf.variables.iteritems():
            if not name in self.dimensions.keys():
                attributes = OrderedDict(((attr, getattr(var, attr))
                                           for attr in var.ncattrs()))
                self.variables[name] = LazyArray(var, attributes=attributes)

            # Mapping of standard_name to name goes into aliases
            alias = getattr(var, 'standard_name', name)
            if alias != name:
                self.aliases[alias] = name

        # Copy global attributes from parent NetCDF file
        self.attributes = OrderedDict(((attr, getattr(self.netcdf, attr))
                                        for attr in self.netcdf.ncattrs()))

        # Each variable is linked with its dimensions and all aliases
        for var in self.variables.values():
            var.metadata.dimensions = OrderedDict(((dim, self.dimensions[dim])
                                                   for dim in var.dimensions))
            var.metadata.aliases = self.aliases

    def __del__(self):
        try:
            self.netcdf.close()
        except:
            pass

    def __repr__(self):
        with PrintOptions(threshold=10, linewidth=120):
            # FIXME: filepath() does not work for MFDataset, keep the pattern?
            #output = ['%s' % self.netcdf.filepath()]
            output = []
            #output = ['%r' % self.__class__]
            output.append('Attributes:')
            for name, attr in self.attributes.iteritems():
                output.append('    %-16s %s' % (name, attr))
            output.append('Variables:')
            for name, var in self.variables.iteritems():
                output.append('    %-16s (%s)' % (name, ', '.join(['%s: %d' % (dim, len(self.netcdf.dimensions[dim])) for dim in var.dimensions])))
            output.append('Coordinates:')
            for name, dim in self.dimensions.iteritems():
                output.append('    %-16s (%s) %s (%s%d)' % (name, dim.dtype, np.asanyarray(dim), 'unlimited, currently: ' if self.unlimited == name else '', dim.size))
        return '\n'.join(output)

    def __getitem__(self, name):
        """Retrieve variables or dimensions by name."""
        for collection in (self.variables, self.dimensions):
            try:
                return collection[name]
            except KeyError:
                pass

            try:
                alias = self.aliases[name]
                return collection[alias]
            except KeyError:
                pass

        raise KeyError(name)

    def __setitem__(self, name, array):
        # TODO: When a dimension already exists, the new one could be automatically renamed <name>_2 etc.
        array = np.asanyarray(array)
        if not isinstance(array, FancyArray):
            raise ValueError('array should be an instance of FancyArray, not %s' % type(array))

        for dimname, dimarray in array.metadata.dimensions.iteritems():
            dim = self.netcdf.dimensions.get(dimname, None)
            if dim is not None:
                if dim.size != dimarray.size:
                    raise ValueError('size of dimension "%s" does not match existing dimension in file (%d != %d)' % (dimname, dimarray.size, dim.size))
            else:
                self.netcdf.createDimension(dimname, None if self.unlimited == dimname else dimarray.size)

            is_time = np.issubdtype(dimarray.dtype, np.datetime64)
            if is_time:
                dimarray = dimarray.astype(DATETYPE).astype(float)

            var = self.netcdf.variables.get(dimname, None)
            if var is not None:
                if (var != dimarray).any():
                    raise ValueError('values of dimension "%s" do not match existing dimension in file' % dimname)
            else:
                dimvar = self.netcdf.createVariable(dimname, dimarray.dtype, (dimname,))
                dimvar[...] = dimarray

                if is_time:
                    # TODO: Like this the time axis is apparently not recognised by Panoply
                    dimvar.units = DATEUNITS
                    dimvar.calendar = 'standard'
                    #dimvar.standard_name = 'time'


        var = self.netcdf.variables.get(name, None)
        if var is None:
            var = self.netcdf.createVariable(name, array.dtype, array.metadata.dimensions.keys(), zlib=True)
            for attr, value in array.metadata.attributes.iteritems():
                setattr(var, attr, value)
        var[...] = array

        # FIXME: this may be (re)moved once close/sync/__del__ are all implemented
        self.netcdf.sync()
