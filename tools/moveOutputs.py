import pandas as pd
import os
from glob import glob
import shutil

def yearsMonths(start_date,end_date):
    if  isinstance (start_date, str):
        pass
    else:
        start_date=str(start_date)
    year_s=start_date[:4]
    month_s=start_date[4:6]
    if  isinstance (end_date, str):
        pass
    else:
        end_date=str(end_date)
    year_e=end_date[:4]
    month_e=end_date[4:6]

    return [[dt.split('-')[0],dt.split('-')[1]] for dt in pd.date_range(f'{year_s}-{month_s}', f'{year_e}-{month_e}',
                          freq='MS').strftime("%Y-%m").tolist()]


base='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class2/v2/*_{year}{month}*nc'
output='/Users/scausio/Dropbox (CMCC)/PycharmData/MVR_delivery/class2/v2'
start_date=202007
end_date=202208

dates=yearsMonths(start_date,end_date)

for date in dates:
    out_month=os.path.join(output,date[0],date[1])
    os.makedirs(out_month,exist_ok=True)
    fs=glob(base.format(year=date[0],month=date[1]))
    print (date[0],date[1],f'contains {len (fs)}files')
    for f in fs:
        try:
            shutil.move(f,out_month)
        except:
            #[shutil.move(f,os.path.join(out_month,os.path.basename(f))) for f in fs]
            inp=input(f'Do you want to overwrite {f} (y/n)')
            if inp=='y':
                shutil.move(f,os.path.join(out_month,os.path.basename(f)))
            else:
                print (f'{f} skipped')

