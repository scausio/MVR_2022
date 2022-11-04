import luigi
from luigi.task import flatten
from luigi.task_register import Register
from util import group_dates
from .targets import OutputDatasetTarget
from netCDF4 import MFDataset, Dataset


# TODO: extend this task to also merge the TimeSeries data
class MergeDataset(luigi.Task):
    start_date = luigi.DateParameter()
    end_date = luigi.DateParameter()
    task = luigi.Parameter()

    @property
    def resources(self):
        # No concurrent writing to the same file
        return {path: 1 for path in flatten(self.output().path)}

    @property
    def task_class(self):
        # resolve task to a class
        return Register.get_task_cls(self.task)

    def requires(self):
        dates = list(group_dates(self.start_date, self.end_date))

        if self.end_date is None or self.start_date == self.end_date:
            # Single day, no grouping
            yield self.task_class(date=start_date)
        else:
            # Group dates and recursively require MetricTimeSeries for smaller chunks
            for start_date, end_date in group_dates(self.start_date, self.end_date or self.start_date):
                if start_date == end_date:
                    yield self.task_class(date=start_date)
                else:
                    yield self.__class__(start_date=start_date, end_date=end_date, task=self.task)

    def output(self):
        return OutputDatasetTarget(self.task_class.metric_id, self.start_date, self.end_date)

    def complete(self):
        variable = self.task_class(date=self.start_date).name
        try:
            data = self.output().open()
            return variable in data.variables
        except:
            return False

    def run(self):
        targets = flatten([task.output() for task in self.requires()])
        datasets = filter(lambda t: isinstance(t, OutputDatasetTarget), targets)
        files = flatten([d.path for d in datasets])

        variable = self.task_class(date=self.start_date).name

        indata = MFDataset(files)
        dimensions = indata.variables[variable].dimensions
        variables = flatten([dimensions, variable])

        # Use the raw Dataset for this operation
        dataset = self.output().open(mode='r+' if self.output().exists() else 'w')
        with dataset.netcdf as outdata:

            for dim in dimensions:
                if not dim in outdata.dimensions:
                    dimension = indata.dimensions[dim]
                    size = None if dimension.isunlimited() else len(dimension)
                    outdata.createDimension(dim, size)

            for var in variables:
                if not var in outdata.variables:
                    variable = indata.variables[var]
                    fill = getattr(variable, '_FillValue', None)

                    outvariable = outdata.createVariable(var, variable[0].dtype if len(variable) else variable.dtype,
                                                         variable.dimensions, fill_value=fill)
                    outvariable[...] = variable[...]

                    for attr in variable.ncattrs():
                        if attr not in ('_FillValue', 'scale_factor', 'add_offset'):
                            setattr(outvariable, attr, getattr(variable, attr))

            for attr in indata.ncattrs():
                setattr(outdata, attr, getattr(indata, attr))
