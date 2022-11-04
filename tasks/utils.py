import yaml
import os
from munch import Munch
import pandas as pd
import time
import subprocess
import numpy as np
from datetime import timedelta

def getJID(stdout):
    return stdout.split()[1][1:-1]

def buildDirTree(workingDir):
    os.makedirs(os.path.join(workingDir, 'observations'),exist_ok=True)
    [os.makedirs(os.path.join(workingDir,'observations',o),exist_ok=True) for o in ['argo','sst','sla','moor_tracer']]
    os.makedirs(os.path.join(workingDir,'logs'),exist_ok=True)
    os.makedirs(os.path.join(workingDir, 'output'), exist_ok=True)
    os.makedirs(os.path.join(workingDir, 'output','result'), exist_ok=True)
    os.makedirs(os.path.join(workingDir, 'output', 'class4'), exist_ok=True)
    os.makedirs(os.path.join(workingDir, 'output', 'class2'), exist_ok=True)

def buildDepCMD(jobIds):
    if isinstance(jobIds,list):
        dn=[f"done({int(i)})" for i in jobIds]
        return ' && '.join(dn)
    else:
        return f"done({int(jobIds)})"


def getConfigurationByID(conf,confId):
    globalConf = yaml.safe_load(open(conf))
    return Munch.fromDict(globalConf[confId])


def getMonthsFirstDay(start_date,end_date):
    return pd.date_range(f"{start_date}"
                  , f"{end_date}",
                  freq='MS').strftime("%Y-%m-%d").tolist()

def getMonthsLastDay(start_date,end_date):
    return pd.date_range(f"{start_date}"
                  , f"{end_date}",
                  freq='M').strftime("%Y-%m-%d").tolist()


def getMonthsFirstDay_minus2(start_date,end_date):
    return (pd.date_range(f"{start_date}", f"{end_date}",freq='M') - pd.DateOffset(2)).strftime("%Y-%m-%d").tolist()#


def getMonthsFirstDay_plus2(start_date,end_date):
    print (start_date,end_date)
    return (pd.date_range(f"{start_date}", f"{end_date}", freq='M')+ pd.DateOffset(2)).strftime("%Y-%m-%d").tolist()

def getMonthsFirst_LastDays(start_date,end_date):
    start=getMonthsFirstDay(start_date,end_date)
    end=getMonthsLastDay(start_date,end_date)
    return list(zip(start,end))

def getNextMonthsFirstDay(start_date,end_date):

    dates=pd.date_range(f"{start_date}"
                  , f"{end_date}",
                  freq='MS')
    return [(i + pd.DateOffset(months=1)).strftime("%Y-%m-%d") for i in dates]

def getMonthsFirst_NextMonthFirstDays(start_date,end_date):   
    start=getMonthsFirstDay(start_date,end_date)
    end=getNextMonthsFirstDay(start_date,end_date)
    return list(zip(start,end))

def getNextMonthsFirstDay_plus2(start_date,end_date):
    dates=pd.date_range(f"{start_date}"
                  , f"{end_date}",
                  freq='MS')
    return [(i + pd.DateOffset(months=1,days=2)).strftime("%Y-%m-%d") for i in dates]

def getMonthsFirst_NextMonthFirstDays_plusminus2(start_date,end_date):
    start=getMonthsFirstDay_minus2(start_date,end_date)
    end=getNextMonthsFirstDay_plus2(start_date,end_date)
    return list(zip(start,end))

class WaitJobEnd():
    def __init__(self, jobIds):
        if not jobIds:
            print ('Nothing to wait')
        else:
            if isinstance(jobIds,list):
                pass
            else:
                jobIds=[jobIds]
            self.jobIds = jobIds
            self.waitJobsEnd()

    def waitJobsEnd(self):
        status = self.checkStatus()
        while not self.allDone(status):
            print('waiting...')
            time.sleep(3)
            status = self.checkStatus()

    def allDone(self, stats):
        stats = np.array(stats)
        return np.all(stats == 'DONE')

    def findIndex(self, splittedList):
        for i, element in enumerate(splittedList):
            if element in ['PEND', 'RUN', 'DONE']:
                # print ('idx:',i)
                return i

    def checkStatus(self):
        status = []
        print (self.jobIds)
        for jobId in self.jobIds:
            s = subprocess.run('bjobs %s' % (jobId), capture_output=True, shell=True, text=True).stdout
            print (s)
            sts = s.split('\n')[1].split(' ')  # [5]
            idx = self.findIndex(sts)
            print(sts[idx])
            status.append(sts[idx])
        return status
