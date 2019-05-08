# This class takes care of loading the Mastodon dialogs from files that contain linearized dialogs from the Mastodon Twitter-like social network
# The required training files are: tmptrain.wds.{0-9}, where each digit represents one cross-validation fold
# The required development files are: tmpdev.wds.{0-9}
# The required validation file is: datatestJoint.idx (there is no cross validation for validation)

# All these files are commited in github along with this code
# There are other files with additional information, such as the .tootids files, which contain a unique identifier from the octodon.social Mastodon server for every post. This identifier allows to retrieve the original post and all of its meta-data from the octodon.social server, but it is not required to train the model in the paper.

corpdir = "../corpus/"

class Tweet():
    def __init__(self,l):
        ss=l.split()
        self.uttid=int(ss[0])
        self.isFirstTweet = self.uttid == 0
        tags=ss[1].split("_")
        self.tagDA=int(tags[1])
        self.tagSE=int(tags[0])
        self.words=[int(x) for x in ss[2:]]

class MastoData():
    def __init__(self,task):
        # "task" may be:
        # "DA" for dialog act recognition only
        # "SE" for sentiment analysis only
        # "JO" for joint dialog act and sentiment recognition
        self.task=task
        self.traindialogs=[]
        self.devdialogs=[]
        # The files tmptrain* must exist
        for i in range(10):
            trainFold=[]
            curdialog=[]
            with open(corpdir+"./tmptrain.wds."+str(i),"r") as f:
                for l in f:
                    tweet = Tweet(l)
                    if tweet.isFirstTweet:
                        if len(curdialog)>0:
                            trainFold.append(curdialog)
                            curdialog=[]
                    curdialog.append(tweet)
                self.traindialogs.append(trainFold)
            devFold=[]
            curdialog=[]
            with open(corpdir+"./tmpdev.wds."+str(i),"r") as f:
                for l in f:
                    tweet = Tweet(l)
                    if tweet.isFirstTweet:
                        if len(curdialog)>0:
                            devFold.append(curdialog)
                            curdialog=[]
                    curdialog.append(tweet)
                self.devdialogs.append(devFold)

        testFold=[]
        curdialog=[]
        with open(corpdir+"/datatestJoint.idx","r") as f:
            for l in f:
                tweet = Tweet(l)
                if tweet.isFirstTweet:
                    if len(curdialog)>0:
                        testFold.append(curdialog)
                        curdialog=[]
                curdialog.append(tweet)
            self.testdialogs=testFold

        self.nvoc = max([max([max([max(t.words) for t in d]) for d in f]) for f in self.devdialogs])+1
        vv = max([max([max([max(t.words) for t in d]) for d in f]) for f in self.traindialogs])+1
        if vv>self.nvoc: self.nvoc=vv
        vv = max([max([max(t.words) for t in d]) for d in self.testdialogs])+1
        if vv>self.nvoc: self.nvoc=vv
        if task=="DA":
            self.nlabs = [max([max([max([t.tagDA for t in d]) for d in f]) for f in self.devdialogs])+1]
        elif task=="SE":
            self.nlabs = [max([max([max([t.tagSE for t in d]) for d in f]) for f in self.devdialogs])+1]
        elif task=="JO":
            self.nlabs = [max([max([max([t.tagDA for t in d]) for d in f]) for f in self.devdialogs])+1, max([max([max([t.tagSE for t in d]) for d in f]) for f in self.devdialogs])+1]
        print("initvoc",self.nvoc,self.nlabs)
    
    def getData(self, part="train", fold=0):
        dials=None
        if part=="train": dials = self.traindialogs[fold]
        elif part=="test": dials = self.testdialogs
        elif part=="dev": dials = self.devdialogs[fold]
        if dials==None: raise

        x,yDA,ySE = [],[],[]
        for dialogidx in range(len(dials)):
            dialx, dialyDA, dialySE = [], [], []
            for tweetidx in range(len(dials[dialogidx])):
                dialx.append(dials[dialogidx][tweetidx].words)
                dialyDA.append(dials[dialogidx][tweetidx].tagDA)
                dialySE.append(dials[dialogidx][tweetidx].tagSE)
            x.append(dialx)
            yDA.append(dialyDA)
            ySE.append(dialySE)
        if self.task=="DA": return x,yDA,None
        elif self.task=="SE": return x,ySE,None
        elif self.task=="JO": return x,yDA,ySE

    def getTest(self):
        return self.getData("test")
    def getTrain(self, fold=0):
        return self.getData("train",fold)
    def getDev(self, fold=0):
        return self.getData("dev",fold)

    def info(self):
        print("nfolds %d" % len(self.traindialogs))
        print("ndialogs %d" % sum([len(d) for d in self.devdialogs]))
        print("nwords %d nLabs %d " % (self.nvoc, self.nlabs))

def evalDA(haty, goldy):
    nclasses = 15
    nok = [0.]*nclasses
    nrec = [0.]*nclasses
    ntot = [0.]*nclasses
    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy]+=1
        nrec[recy]+=1
        if recy==gldy: nok[gldy]+=1

    nsamps = sum(ntot)
    preval=[float(ntot[i])/float(nsamps) for i in range(nclasses)]
    prec=0.
    reca=0.
    raweval="DAraweval "
    for j in range(nclasses):
        tp = nok[j]
        pr,re = 0.,0.
        if nrec[j]>0: pr=float(tp)/float(nrec[j])
        if ntot[j]>0: re=float(tp)/float(ntot[j])
        raweval+=str(pr)+"_"+str(re)+" "
        prec += pr*preval[j]
        reca += re*preval[j]
    print(raweval)
    if prec+reca==0.: f1=0.
    else: f1 = 2.*prec*reca/(prec+reca)
    return f1

def evalSE(haty, goldy):
    nclasses = 3
    nok = [0.]*nclasses
    nrec = [0.]*nclasses
    ntot = [0.]*nclasses
    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy]+=1
        nrec[recy]+=1
        if recy==gldy: nok[gldy]+=1

    raweval="SEraweval "
    f1pos,f1neg=0.,0.
    for j in (1,): # 1=+ and 2=-
        tp = nok[j]
        pr,re = 0.,0.
        if nrec[j]>0: pr=float(tp)/float(nrec[j])
        if ntot[j]>0: re=float(tp)/float(ntot[j])
        raweval+=str(pr)+"_"+str(re)+" "
        if pr+re>0.: f1pos = 2.*pr*re/(pr+re)
    for j in (2,): # 1=+ and 2=-
        tp = nok[j]
        pr,re = 0.,0.
        if nrec[j]>0: pr=float(tp)/float(nrec[j])
        if ntot[j]>0: re=float(tp)/float(ntot[j])
        raweval+=str(pr)+"_"+str(re)+" "
        if pr+re>0.: f1neg = 2.*pr*re/(pr+re)
    print(raweval)
    f1=(f1pos+f1neg)/2.
    return f1

if __name__ == "__main__":
    corpus = MastoData()
    corpus.info()

