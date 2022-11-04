import subprocess
import os
import time
from tasks.utils import getMonthsFirst_LastDays,buildDepCMD,getJID

class Process():
    def __init__(self,conf,start_date,end_date,deps=False):
        self.conf=conf
        self.start_date=start_date
        self.end_date=end_date
        self.deps = deps

    def iterateDate(self, cmd, obs_type=False):
        """
        months=getMonthsFirst_LastDays(self.start_date,self.end_date)
        print (months)
        ids=[]
        for month in months:
            print (f'Running PQ for {month[0]} {obs_type}')
            cm=cmd(month,obs_type)
            print (cm)
            if not os.path.exists(f"{(cm.split('-o')[-1]).split('.')[0]}.nc"[1:]):
                print (f" {(cm.split('-o')[-1]).split('.')[0]}.nc"[1:])
                id = subprocess.run(cm, capture_output=True, shell=True, text=True).stdout
                ids.append(id.replace('\n',''))
        print(ids)
        return ids
        """
        print (f'Running PQ for {self.start_date} {obs_type}')
        cm=cmd((self.start_date,self.end_date),obs_type)
        print (cm)
        if not os.path.exists(f"{(cm.split('-o')[-1]).split('.')[0]}.nc"[1:]):
            print (f" {(cm.split('-o')[-1]).split('.')[0]}.nc"[1:])
            id = getJID(subprocess.run(cm, capture_output=True, shell=True, text=True).stdout)
            return id
        else:
            return False

class OProcess(Process):
    def cmd(self,date,obs_type):
        start_date, end_date=date
        obs_name=obs_type.split("_")[0]
        if obs_name=='moor':
            obs_name='moor_extract'
        proc_path=os.path.join(self.conf.pq_path, f'bin/process_{obs_name}.py')
        cat_path=os.path.join(self.conf.pq_path,f'pq_catalogs/catalog.yaml')
        ref_period=f"{start_date.split('-')[0]}{start_date.split('-')[1]}"
        outfile=os.path.join(self.conf.interm_path,obs_type,f"{obs_type}_{ref_period}.nc")

        if self.deps:
            dep=f"-w \"{buildDepCMD(self.deps)}\" "
        else:
            dep=''
        time.sleep(2)
        return f"bsub -J {obs_type} {dep} -P {self.conf.proj_code}  -M 15G -R \"rusage[mem=15G]\" " \
               f"-o {self.conf.logs}/{obs_type}_{ref_period}_%J.out -e {self.conf.logs}/{obs_type}_{ref_period}_%J.err " \
            f"python {proc_path} " \
            f"-c {cat_path} " \
            f"-n {obs_type} " \
            f"-s {start_date} -e {end_date} " \
            f"-o {outfile}"

    def argo(self):
        return self.iterateDate(self.cmd, 'argo')

    def sla(self):
        return self.iterateDate(self.cmd, 'sla')

    def sst(self):
        return self.iterateDate(self.cmd, 'sst')

    def moor_tracer(self):
        return self.iterateDate(self.cmd, 'moor_tracer')


class MProcess(Process):
    def __init__(self,conf,start_date,end_date,exp_name,validation_class,deps=False):
        Process.__init__(self,conf,start_date,end_date, deps)
        self.exp_name=exp_name
        self.val_cl=validation_class
    def cmd(self,date,obs_type):
        start_date, end_date = date
        ref_period = f"{start_date.split('-')[0]}{start_date.split('-')[1]}"
        interm = os.path.join(self.conf.interm_path, obs_type, f'{obs_type}_{ref_period}.nc')
        if obs_type == 'argo':
            interp='interpolate_model.py'
        elif obs_type == 'sla':
            interp = 'interpolate_model_SLA.py'
        elif obs_type == 'moor_tracer':
            interp = 'extract_model_Moor.py'
            interm = os.path.join(self.conf.interm_path, obs_type, f'{obs_type}_all.nc')
        else:
            interp = 'interpolate_model_2D.py'
        proc_path=os.path.join(self.conf.pq_path, f'bin/{interp}')
        cat_path=os.path.join(self.conf.pq_path,f'pq_catalogs/catalog.yaml')

        outpath=os.path.join(self.conf.output,ref_period)
        os.makedirs(outpath,exist_ok=True)
        outfile=os.path.join(outpath,f'{self.val_cl}_{obs_type}_{self.exp_name}_{ref_period}.nc')
        if self.deps:
            dep=f"-w \"{buildDepCMD(self.deps)}\" "
        else:
            dep=''
        #time.sleep(2)
        return f"bsub -J MOD_{obs_type} {dep} -P {self.conf.proj_code} -M 30G -R \"rusage[mem=30G]\" " \
               f"-o {self.conf.logs}/{self.exp_name}_{obs_type}_{ref_period}_%J.out -e {self.conf.logs}/{self.exp_name}_{obs_type}_{ref_period}_%J.err " \
            f"python {proc_path} " \
            f"-c {cat_path} " \
            f"-s {start_date} -e {end_date} " \
            f"-n {self.exp_name} " \
            f"-i {interm} " \
            f"-vc {self.val_cl} " \
            f"-o {outfile}"

    def argo(self):
        return self.iterateDate(self.cmd, 'argo')

    def sla(self):
        return self.iterateDate(self.cmd, 'sla')

    def sst(self):
        return self.iterateDate(self.cmd, 'sst')

    def moor_tracer(self):
        return self.iterateDate(self.cmd, 'moor_tracer')


class IProcess(Process):
    def __init__(self,conf,start_date,end_date,exp_name,validation_class,deps=False):
        Process.__init__(self,conf,start_date,end_date, deps)
        self.exp_name=exp_name
        self.val_cl=validation_class
    def cmd(self,date,obs_type):
        start_date, end_date = date
        ref_period = f"{start_date.split('-')[0]}{start_date.split('-')[1]}"
        proc_path=os.path.join(self.conf.pq_path, f'bin/process_intermediate.py')
        conf_path='conf.yaml'
        if self.deps:
            dep=f"-w \"{buildDepCMD(self.deps)}\" "
        else:
            dep=''
        #time.sleep(2)
        return f"bsub -J PINT_{obs_type} {dep} -P {self.conf.proj_code} -M 1G -R \"rusage[mem=1G]\" " \
               f"-o {self.conf.logs}/PINT_{self.exp_name}_{obs_type}_{ref_period}_%J.out -e {self.conf.logs}/PINT_{self.exp_name}_{obs_type}_{ref_period}_%J.err " \
            f"python {proc_path} " \
            f"-c {conf_path} " \
            f"-d {ref_period} " \
            f"-m {self.exp_name} " \
            f"-o {obs_type} " \
            f"-vc {self.val_cl}"

    def argo(self):
        return self.iterateDate(self.cmd, 'argo')

    def sla(self):
        return self.iterateDate(self.cmd, 'sla')

    def sst(self):
        return self.iterateDate(self.cmd, 'sst')

    def moor_tracer(self):
        return self.iterateDate(self.cmd, 'moor_tracer')


class PostProc(Process):
    def __init__(self,conf,start_date,end_date,validation_class,deps=False):
        Process.__init__(self,conf,start_date,end_date, deps)
        self.val_cl=validation_class
        self.run()
    def cmd(self,date,obs_type):
        start_date, end_date=date
        proc_path=os.path.join(self.conf.pq_path, f'bin/post_processing.py')
        conf_path='conf.yaml'

        if self.deps:
            dep=f"-w \"{buildDepCMD(self.deps)}\" "
        else:
            dep=''
        return f"bsub -J POST_{self.val_cl} {dep} -P {self.conf.proj_code}  -M 1G -R \"rusage[mem=1G]\" " \
               f"-o {self.conf.logs}/POST_{self.val_cl}_{start_date[:7]}_%J.out -e {self.conf.logs}/POST_{self.val_cl}_{start_date[:7]}_%J.err " \
            f"python {proc_path} " \
            f"-c {conf_path} " \
            f"-s {start_date.replace('-','')} -e {end_date.replace('-','')} " \
            f"-vc {self.val_cl} "

    def run(self):
        return self.iterateDate(self.cmd)

