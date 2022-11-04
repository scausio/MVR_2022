#!/usr/bin/env python
from sys import argv
from calendar import monthrange
from tasks.utils import getMonthsFirst_LastDays, getMonthsFirst_NextMonthFirstDays, \
    getMonthsFirst_NextMonthFirstDays_plusminus2, getJID
import subprocess
from tasks.utils import buildDepCMD


def main():
    from_year_month = argv[1]
    try:
        to_year_month = argv[2]
    except:
        to_year_month = argv[1]
    if (len(from_year_month) > 6) or (len(to_year_month) > 6):
        exit('Date not set properly. Please use YYYYMM')
    start_date = f"{from_year_month}01"
    end_date = f'{to_year_month}{monthrange(int(to_year_month[:4]), int(to_year_month[4:]))[1]:02d}'
    print(f'Running MVR from {start_date} to {end_date}')
    months = getMonthsFirst_NextMonthFirstDays(start_date, end_date)
    print(months)
    ids = []
    for i, month in enumerate(months):
        print(f'Running {month[0]} ')
        if i == 0:
            cm = f'bsub -q s_long -J main{month[0]} -P 0512 python main.py {month[0]} {month[1]}'
        else:
            print(ids)
            dep = f"-w \"{buildDepCMD(ids)}\" "
            cm = f"bsub -q s_long -J main{month[0]} -P 0512 {dep} python main.py {month[0]} {month[1]}"
        print(cm)
        id = getJID(subprocess.run(cm, capture_output=True, shell=True, text=True).stdout)
        ids.append(id)


if __name__ == '__main__':
    main()
