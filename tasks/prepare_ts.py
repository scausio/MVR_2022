import yaml, os
from munch import munchify
import datetime
from glob import glob
from argparse import ArgumentParser
from tasks.utils import getConfigurationByID
import xarray as xr
import shutil


def main(start_date, end_date, cfg, output, grid):
    for fc in cfg.forecasts:
        print(fc)
        d = start_date
        while d <= end_date:
            if cfg.forecasts[fc].fcday != 'an':
                p_date = d - datetime.timedelta(days=int(cfg.forecasts[fc].fcday))
                file_name = cfg.forecasts[fc].filename.format(date=d.strftime("%Y%m%d"),
                                                              p_date=p_date.strftime("%Y%m%d"), grid=grid)

            else:
                print("an")
                p_date = d
                file_name = cfg.forecasts[fc].filename.format(date=(d - datetime.timedelta(days=2)).strftime("%Y%m%d"),
                                                              p_date=p_date.strftime("%Y%m%d"), grid=grid)
            src = os.path.join(cfg.forecasts[fc].in_dir.format(p_date=p_date.strftime("%Y%m%d")), file_name)
            for src in glob(src):
                out_dir = cfg.forecasts[fc].out_dir.format(output=output).replace('ts_cl2', 'ts_cl2_link')

                dst = os.path.join(out_dir, os.path.basename(src))
                print(dst)
                if not os.path.exists(dst):
                    print(dst)
                    os.makedirs(out_dir, exist_ok=True)
                    print(src + "->" + dst)
                    os.symlink(src, dst)
            d = d + datetime.timedelta(days=1)


if __name__ == "__main__":
    parser = ArgumentParser(description='Prepare Forecasts Timeseries')
    parser.add_argument('-c', '--config', default='../example/config_class4.yaml', help='configuration file')
    parser.add_argument('-s', '--start_date', help='start date')
    parser.add_argument('-e', '--end_date', help='end date')
    parser.add_argument('-vt', '--validation_type', help='Class_4 or Class_2')

    grids = ['t', 'u', 'v']

    args = parser.parse_args()

    config = args.config
    start_date = args.start_date
    end_date = args.end_date
    vt = args.validation_type
    cfg = getConfigurationByID('conf.yaml', vt)  # munchify(yaml.load(open(config), Loader=yaml.FullLoader))
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=2)
    start_month = datetime.datetime.strftime(start_date, '%Y%m')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
    output = os.path.join(getConfigurationByID('conf.yaml', 'base').output)

    for grid in grids:
        print(f'processing {grid} grid')
        main(start_date, end_date, cfg, output, grid)

    print('Soft links completed')

    if vt == 'Class_2':
        for fc in cfg.forecasts:
            links = cfg.forecasts[fc].out_dir.format(output=output).replace('ts_cl2', 'ts_cl2_link')
            archive = cfg.forecasts[fc].out_dir.format(output=output).replace('ts_cl2', 'ts_cl2_arch')
            out_dir = cfg.forecasts[fc].out_dir.format(output=output)

            os.makedirs(out_dir, exist_ok=True)
            catalog = getConfigurationByID(
                os.path.join(getConfigurationByID('conf.yaml', 'base').pq_path, 'pq_catalogs', 'catalog.yaml'),
                'sources')[
                f'cl2_{fc}'].metadata.variables

            print(fc)
            tfiles = glob(os.path.join(links, f't_{start_month}*'))
            tfiles_basename = [os.path.basename(f) for f in tfiles]
            tfiles_all = glob(os.path.join(out_dir, f't_*'))
            if tfiles_all:
                for t in tfiles_all:
                    if os.path.basename(t) not in tfiles_basename:
                        shutil.move(t, os.path.join(archive, os.path.basename(t)))
            print(os.path.join(links, f't_{start_month}*'))
            for tfile in tfiles:
                outfile = os.path.join(out_dir, os.path.basename(tfile))
                archfile = os.path.join(archive, os.path.basename(tfile))
                if not os.path.exists(outfile):
                    if not os.path.exists(archfile):
                        print(f" processing {tfile}")
                        ds_t = xr.open_dataset(tfile)
                        ds_u = xr.open_dataset(tfile.replace('t_', 'u_'))[catalog.u]
                        ds_v = xr.open_dataset(tfile.replace('t_', 'v_'))[catalog.v]
                        ds_t[catalog.u] = ((ds_t[catalog.temperature].copy() * 0) + ds_u.values)
                        ds_t[catalog.v] = ((ds_t[catalog.temperature].copy() * 0) + ds_v.values)
                        ds_t.to_netcdf(outfile)
                    else:
                        print(f" coping  {archfile} to {outfile}")
                        shutil.copy(archfile, outfile)
