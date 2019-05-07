# This program loads a set of logfiles produced by launch.sh and compute the average accuracies

import sys
from os import walk
import numpy as np

logdir = "./"
nepoks = 500

def loadlogs():
    print("logdir",logdir)
    fichs=[]
    for (dirpath,dirnames,filenames) in walk(logdir): 
        fichs.extend(filenames)
        break
    res = [f for f in fichs if f.startswith("logdase")]
    return res

class Accs:
    def __init__(self):
        self.accs={}
        self.targettask=0
    def add(self,part,fold,trsize,epok,acc):
        k = (part,trsize)
        if k not in self.accs: self.accs[k]={}
        aa = self.accs[k]
        if fold not in aa: aa[fold]={}
        aaa = aa[fold]
        # allow for multiple runs of same xp
        if epok not in aaa: aaa[epok]=[]
        aaa[epok].append(acc)

    def getTrsize(self):
        ts=set()
        for part,trsize in self.accs.keys(): ts.add(trsize)
        return ts

    def getAccPerEpok(self,trsize):
        k = ("dev",trsize)
        accs = self.accs[k]
        devaccavg = None
        for fold in accs.keys():
            if devaccavg==None:
                devaccavg={}
                # fold X epok X run X task
                for e in range(nepoks):
                    devaccavg[e] = [] # list for tasks, not for runs
                    for task in range(2):
                        vmax=float(accs[fold][e][0][task])
                        for run in range(1,len(accs[fold][e])):
                            # The maximum always choose the best run on the dev corpus: this is to filter out badly initialized runs that may not converge
                            if accs[fold][e][run][task]>vmax: vmax=accs[fold][e][run][task]
                        devaccavg[e].append(vmax)
            else: 
                for e in range(nepoks):
                    tmpdevaccavg = [] # list for tasks, not for runs
                    for task in range(2):
                        vmax=float(accs[fold][e][0][task])
                        for run in range(1,len(accs[fold][e])):
                            if accs[fold][e][run][task]>vmax: vmax=accs[fold][e][run][task]
                        tmpdevaccavg.append(vmax)
                    tmp = [devaccavg[e][task]+tmpdevaccavg[task] for task in range(2)]
                    devaccavg[e]=tmp
        nfolds = len(accs.keys())
        print("nfolds",nfolds)
        for e in range(nepoks):
            tmp = [ac/float(nfolds) for ac in devaccavg[e]]
            devaccavg[e]=tmp
        for epok in range(len(devaccavg)): print("DEVACCEPOK",epok,devaccavg[epok])
        # We now select the best epoch on the developement corpus for the target task:
        bestepok = np.argmax([devaccavg[e][self.targettask] for e in range(nepoks)])
        print("bestepok",bestepok)

        # Next we extract the test results corresponding to this best epoch
        # we could take only one of these results, but we average the test acc over 10 runs
        k = ("test",trsize)
        accs = self.accs[k]
        s=None
        for fold in accs.keys():
            # accs = (fold X epoks X runs X tasks)
            a=accs[fold][bestepok]
            if s==None:
                s=[] # per task
                for task in range(2):
                    # We don't need the following lines / the max(), because all F1s of all runs must be the same on the test
                    vmax=a[0][task]
                    for run in range(len(a)):
                        if a[run][task]>vmax: vmax=a[run][task]
                    s.append(vmax)
            else: 
                tmps=[] # per task
                for task in range(2):
                    vmax=a[0][task]
                    for run in range(len(a)):
                        if a[run][task]>vmax: vmax=a[run][task]
                    tmps.append(vmax)
                tmp = [s[task]+tmps[task] for task in range(2)]
                s=tmp
        for i in range(len(s)): s[i]/=float(len(accs.keys()))
        return s

def analyze(tgt):
    res = loadlogs()
    allaccs = Accs()
    allaccs.targettask=tgt

    for f in res:
        with open(logdir+"/"+f,"r") as g: lines=g.readlines()
        # first line contains the parameters
        parms = lines[0].split()
        fold = int(parms[1])
        lr = float(parms[3])
        embed = int(parms[5])
        hsize = int(parms[7])
        trsize = int(parms[11])
        for s in lines:
            part="none"
            if s.startswith("eval"): part="dev"
            elif s.startswith("test"): part="test"
            if part!="none":
                epok =s.split()[1]
                if epok=="init": epok=-1
                else: epok=int(epok)
                F1DA=float(s.split()[2])
                F1SE=float(s.split()[3])
                allaccs.add(part,fold,trsize,epok,[F1DA,F1SE])

    ts = list(allaccs.getTrsize())
    ts.sort()
    for t in ts:
        print("TESTACC "+str(t)+" "+str(allaccs.getAccPerEpok(t)))

if __name__ == "__main__":
    import sys
    logdir = sys.argv[1]
    targettask=0
    if len(sys.argv)>=3: targettask=int(sys.argv[2])
    analyze(targettask)

