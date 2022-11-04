"""
Sometimes Luigi needs the help of Mario to get things done. ;-)
"""

from os.path import getmtime
from luigi import LocalTarget
from luigi.task import flatten


class InputUpdateMixin(object):
    """
    Mixin that also marks a task incomplete if a dependency was been updated.
    """
    def complete(self):
        if not getattr(self, '_complete', False):
            output = flatten(self.output())
            requires = flatten([task.output() for task in flatten(self.requires())])

            if not all(target.exists() for target in output):
                return False

            elif getattr(self, 'check_input_update', True):
                mtime = min(getmtime(path) for target in output
                                           for path in flatten(getattr(target, 'path', [])))

                for task in flatten(self.requires()):
                    if not task.complete():
                        return False
                        
                    for target in flatten(task.output()):
                        if any(getmtime(path) > mtime for path in flatten(getattr(target, 'path', []))):
                            return False

            # Cache a positive answer to speed up subsequent lookups
            self._complete = True

        return self._complete 

