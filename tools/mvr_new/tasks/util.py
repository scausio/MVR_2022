import collections
from mercator.ticker import DegreeFormatter
from datetime import date, timedelta
from calendar import monthrange


lat_formatter = DegreeFormatter(labels=('S', 'N'))
lon_formatter = DegreeFormatter(labels=('W', 'E'))


def date_range(start_date, end_date=None):
    end_date = end_date or start_date
    date = start_date
    while date <= end_date:
        yield date
        date += timedelta(days=1)


def group_dates(start_date, end_date):
    """
    Break up the period between start_date, end_date in ranges of full years,
    full months and leave as few individual days as possible
    """
    date = start_date

    while date <= end_date:
        end_month = date.replace(day=monthrange(date.year, date.month)[1])
        end_year = date.replace(month=12, day=31)

        if date.day > 1 or end_month >= end_date:
            yield date, date
            date += timedelta(days=1)
        elif date.month > 1 or end_year >= end_date:
            yield date, end_month
            date = end_month + timedelta(days=1)
        else:
            yield date, end_year
            date = end_year + timedelta(days=1)


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

