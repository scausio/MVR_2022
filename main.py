#!/usr/bin/env python
import os
import subprocess
from tasks.utils import getConfigurationByID,buildDirTree,getMonthsFirst_LastDays,WaitJobEnd,getJID
from tasks.pq.processing import OProcess, MProcess, IProcess,PostProc
from sys import argv
import time

class DepsManager():
    def __init__(self):
        self.deps=[]
    def check(self,deps,key):
        glbs=globals()
        if key in glbs:
            key=glbs[key]
            if isinstance(key, list):
                [deps.append(JId) for JId in key if JId]
            else:
                if key:
                    deps.append(key)
        else:
            print(f'{key} not exists')
        return deps



    def observations(self,deps):
        argo='ARGO_JId'
        sst='SST_JId'
        sla='SLA_JId'
        moor='MOORtr_JId'
        models_list=[argo,sst,sla,moor]
        for model in models_list:
            deps=self.check(deps,model)
        return deps

    def models(self,deps):
        argo = 'IARGO_JId'
        sst = 'ISST_JId'
        sla = 'ISLA_JId'
        moor = 'IMOOR_JId'
        interms_list = [argo, sst, sla, moor]
        for interm in interms_list:
            deps=self.check(deps,interm)
        return deps

    def intermediates(self,deps):
        argo = 'PINT_ARGO_JId'
        sst = 'PINT_SST_JId'
        sla = 'PINT_SLA_JId'
        moor = 'PINT_MOOR_JId'
        posts_list = [argo, sst, sla,moor]
        for post in posts_list:
            deps=self.check(deps,post)
        return deps

    def timeseries(self, deps):
        ts2 = 'TS_C2_JId'
        ts4 = 'TS_C4_JId'
        ts_lists=[ts2,ts4]
        for ts in ts_lists:
            deps = self.check(deps, ts)
        return deps

start_date=argv[1]
end_date=argv[2]

subprocess.call('setup.sh',shell=True)
config=getConfigurationByID('conf.yaml','base')
buildDirTree(os.getcwd())

proj=config.proj_code
logToday=config.logs
depsManager = DepsManager()

# 1. Prepare Timeseries
TS_C4_Jcmd=f"bsub -q s_short -J TS_C4 -P {proj} -o {logToday}/TS_C4_{start_date}_%J.out -e {logToday}/TS_C4_${start_date}_%J.err python ./tasks/prepare_ts.py -vt Class_4 -s {start_date} -e {end_date}"
TS_C2_Jcmd=f"bsub  -q s_long -J TS_C2 -P {proj} -o {logToday}/TS_C2_{start_date}_%J.out -e {logToday}/TS_C2_${start_date}_%J.err  python ./tasks/prepare_ts.py -vt Class_2 -s {start_date} -e {end_date}"
print(TS_C2_Jcmd)

deps = depsManager.deps
TS_C4_JId=getJID(subprocess.run(TS_C4_Jcmd, capture_output=True, shell=True, text=True).stdout)
TS_C2_JId=getJID(subprocess.run(TS_C2_Jcmd, capture_output=True, shell=True, text=True).stdout)
deps = depsManager.timeseries(deps)
print (deps)
WaitJobEnd(deps)

#
# #
# # # Class 4
# # # 1.1 Process Intermediate Observations
deps = depsManager.deps
ARGO_JId=OProcess(config, start_date,end_date).argo()
SST_JId=OProcess(config, start_date,end_date).sst()
SLA_JId=OProcess(config, start_date,end_date).sla()
# MOORtr_JId=OProcess(config, start_date,end_date).moor_tracer()
deps = depsManager.observations(deps)
print(f'waiting {deps} for Observations Processing')
WaitJobEnd(deps)

# Process model
deps = depsManager.deps
for idx in config.forecast_idxs:
    IARGO_JId = MProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').argo()
    ISST_JId = MProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').sst()
    ISLA_JId = MProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').sla()

    IMOOR_JId = MProcess(config, start_date, end_date, f"fc_{idx}", 'cl2').moor_tracer()
    print(IMOOR_JId)
    deps = depsManager.models(deps)
    print (deps)
print(f'waiting {deps} for Model Processing')
WaitJobEnd(deps)

# # process intermediate
deps = depsManager.deps
for idx in config.forecast_idxs:
    PINT_ARGO_JId = IProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').argo()
    PINT_SST_JId = IProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').sst()
    PINT_SLA_JId = IProcess(config, start_date, end_date, f"fc_{idx}", 'cl4').sla()

    PINT_MOOR_JId = IProcess(config, start_date, end_date, f"fc_{idx}", 'cl2').moor_tracer()
    deps = depsManager.intermediates(deps)
print(f'waiting {deps} for Intermediate Processing')
WaitJobEnd(deps)


CL4POST_JID=PostProc(config, start_date, end_date, 'Class_4')
#CL2POST_JID=PostProc(config, start_date, end_date, 'Class_2')
CL2POST_JID=PostProc(config, start_date, end_date, 'Class_2', PINT_MOOR_JId)
time.sleep(5)

print (start_date, 'complete')

WaitJobEnd(CL2POST_JID)
