import random
import sys

class Dialog:
    def __init__(self):
        self.ids=[] # tweet ID for all segments in the dialog
        self.wds=[]
        self.yda=[]
        self.yse=[]

    def addline(self,l,ls):
        s=ls.split()
        self.ids.append(s[-1])
        s=l.split()
        tags=s[1]
        twotags = tags.split("_")
        self.yda.append(int(twotags[1]))
        self.yse.append(int(twotags[0]))
        self.wds.append([int(x) for x in s[2:]])

    def ids2str(self):
        s=""
        for tootid in self.ids: s+=str(tootid)+'\n'
        return s

    def tostr(self):
        s=""
        for i in range(len(self.wds)): s+=str(i)+" "+str(self.yse[i])+'_'+str(self.yda[i])+" "+' '.join([str(x) for x in self.wds[i]])+'\n'
        return s

suffix = sys.argv[1]

with open("../corpus/datatrainJoint.idx","r") as f: lines=f.readlines()
with open("../corpus/datatrainJoint.txt","r") as f: lines2=f.readlines()
assert len(lines)==len(lines2)

# build all dialogs from train, with their tweet ids
ds=[]
d=Dialog()
i=0
while i<len(lines):
    if lines[i][0]=='0':
        if len(d.ids)>0: ds.append(d)
        d=Dialog()
    d.addline(lines[i],lines2[i])
    i+=1

# split train/dev on the first tweet id, just like for train/test (see corpus.py)
firstID = list(set([d.ids[0] for d in ds]))
random.shuffle(firstID)
ntrain=int(float(len(firstID))*0.9)
trainIDs = firstID[:ntrain]
devIDs = firstID[ntrain:]
dstrain = [d for d in ds if d.ids[0] in trainIDs]
dsdev   = [d for d in ds if d.ids[0] in devIDs]

with open("../corpus/tmptrain.tootids."+suffix,"w") as f:
    for d in dstrain: f.write(d.ids2str())
with open("../corpus/tmpdev.tootids."+suffix,"w") as f:
    for d in dsdev: f.write(d.ids2str())

with open("../corpus/tmptrain.wds."+suffix,"w") as f:
    for d in dstrain: f.write(d.tostr())
with open("../corpus/tmpdev.wds."+suffix,"w") as f:
    for d in dsdev: f.write(d.tostr())

