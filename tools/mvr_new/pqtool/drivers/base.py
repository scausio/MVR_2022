import intake
import numpy as np
from datetime import datetime
from glob import glob


class DataSource(intake.source.base.DataSource, intake.source.base.PatternMixin):

    def __init__(self, urlpath, metadata=None, path_as_pattern=True):
        self.path_as_pattern = path_as_pattern
        self.urlpath = urlpath

        if isinstance(self.urlpath, list):
            self.files = self.urlpath
        else:
            self.files = sorted(glob(self.urlpath))

        super(DataSource, self).__init__(metadata=metadata)

    def _get_schema(self):
        return intake.source.base.Schema(
            datashape=None,
            dtype=None,
            shape=(None, 2),
            npartitions=len(self.files),
            extra_metadata={}
        )

    def subset(self, **kwargs):
        files = np.array(self.files)
        match = np.ones_like(files, dtype=bool)
        for field, values in intake.source.utils.reverse_formats(self.pattern, files).items():
            values = np.asanyarray(values)
            if values.size and field in kwargs:
                search = kwargs[field]

                if isinstance(values[0], datetime):
                    def converter(date):
                        return np.datetime64(date).astype('datetime64[ms]').astype(datetime)

                    if isinstance(search, slice):
                        search = slice(converter(search.start), converter(search.stop))
                    elif isinstance(search, list):
                        search = list(map(converter, search))
                    else:
                        search = converter(search)

                if isinstance(search, slice):
                    if search.start is not None:
                        match = np.logical_and(match, values >= search.start)
                    if search.stop is not None:
                        match = np.logical_and(match, values < search.stop)

                elif isinstance(search, list):
                    submatch = np.zeros_like(match)
                    for s in search:
                        submatch = np.logical_or(submatch, values == s)
                    match = np.logical_and(match, submatch)
                else:
                    match = np.logical_and(match, values == search)

        ds = self.__class__(urlpath=files[match].tolist(),
                            metadata=self.metadata,
                            path_as_pattern=True)

        ds.name = self.name
        ds._original_urlpath = self._original_urlpath or self.urlpath
        return ds
