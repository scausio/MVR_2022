import yaml, os
from munch import munchify
import datetime
import glob
from argparse import ArgumentParser

parser = ArgumentParser(description='Prepare Forecasts Timeseries')
parser.add_argument('-c', '--config', default='../example/config_class4.yaml', help='configuration file')
parser.add_argument('-s', '--start_date', help='start date')
parser.add_argument('-e', '--end_date', help='end date')

args = parser.parse_args()

config = args.config
start_date = args.start_date
end_date = args.end_date
cfg = munchify(yaml.load(open(config), Loader=yaml.FullLoader))

start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
end_date = datetime.datetime.strptime(end_date, '%Y%m%d')

for fc in cfg.forecasts:
    d = start_date
    while d <= end_date:
        if cfg.forecasts[fc].fcday != 'an':
            p_date = d - datetime.timedelta(days=int(cfg.forecasts[fc].fcday))
        else:
            print("an")
            p_date = d
        file_name = cfg.forecasts[fc].filename.format(date=d.strftime("%Y%m%d"), p_date=p_date.strftime("%Y%m%d"))
        src = cfg.forecasts[fc].in_dir.format(year=str(d.year), month=str(d.month).zfill(2), p_date=p_date.strftime("%Y%m%d")) + file_name
        print(src)
        for src in glob.glob(src):
            print(src)
            dst = cfg.forecasts[fc].out_dir + os.path.basename(src)
            print(src + "->" + dst)
            os.symlink(src, dst)
        d = d + datetime.timedelta(days=1)


