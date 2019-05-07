import sys
import numpy as np
import codecs
import torch
from torch.autograd import Variable
import torch.nn as nn
from random import shuffle
from data import evalDA
from data import evalSE

# These two functions are just baseline performances, when answering the most frequent label or a random label

def printMostFrequent(goldy1,goldy2):
    from collections import Counter
    flat_list = [item for sublist in goldy1 for item in sublist]
    co=Counter(flat_list)
    yy=co.most_common(1)[0][0]
    recys = [yy]*len(flat_list)
    f1=evalDA(recys,flat_list)
    print("Mostfrequent DA: %f" % f1)
    flat_list = [item for sublist in goldy2 for item in sublist]
    co=set(flat_list)
    s="All SE: "
    for yy in co:
        recys = [yy]*len(flat_list)
        f1=evalSE(recys,flat_list)
        s+=str(f1)+" "
    print(s)

def printRandom(goldy1,goldy2):
    flat_list = [item for sublist in goldy1 for item in sublist]
    nlabs = max(flat_list)+1
    recys = np.random.randint(0,nlabs,size=len(flat_list))
    f1=evalDA(recys,flat_list)
    print("Random DA: %f %d" % (f1,nlabs))
    flat_list = [item for sublist in goldy2 for item in sublist]
    nlabs = max(flat_list)+1
    recys = np.random.randint(0,nlabs,size=len(flat_list))
    f1=evalSE(recys,flat_list)
    print("Random SE: %f %d" % (f1,nlabs))

class Model(nn.Module):
    def __init__(self, nvoc, nlabs, embed, hsize=100, lr=0.001, nepok=100, useRNN=True, limitDA=-1, limitSE=-1):
        super(Model,self).__init__()

        self.limitDA=limitDA
        self.limitSE=limitSE
        print("MODCONFIG",nvoc,nlabs,embed,hsize,lr,nepok,useRNN,limitDA,limitSE)

        self.nlabs = nlabs
        self.embedsize = embed
        self.h = hsize
        self.nepochs = nepok
        self.lr = lr

        self.embed = nn.Embedding(nvoc,self.embedsize)
        self.dropout = nn.Dropout(0.4)
        self.uttlstm = nn.LSTM(input_size=self.embedsize,hidden_size=self.h,num_layers=1,bidirectional=True,batch_first=True)
        self.dialrnn = nn.RNN(input_size=self.h*2,hidden_size=self.h,num_layers=1,bidirectional=False,batch_first=True,nonlinearity='relu')
        if useRNN: inmlpsize = self.h
        else: inmlpsize = self.h * 2
        # one MLP per task
        self.mlps = nn.ModuleList([nn.Linear(inmlpsize,nl) for nl in nlabs])
        self.useRNN=useRNN
        self.usecuda=False

    # the model is agnostic to whether we have DA,SE or SE,DA or ...
    def forward(self,x):
        # X is a list of list of Variable = Nsamps x Nutts x Nwords
        out1 = []
        out2 = []
        # naive implementation
        nsamps = len(x)
        for samp in range(nsamps):
            nutts  = len(x[samp])
            rnndialinputs=[]
            for utt in range(nutts):
                nwords = x[samp][utt].data.size()[0]
                e = self.embed(x[samp][utt]).view(1,nwords,-1)
                # got batch=1 X time=nwords X embeddims
                uttout,_ = self.uttlstm(e)
                # uttout = self.dropout(uttout)
                uttout = nn.functional.relu(uttout)
                # got batch X time X hsize*ndirs
                # use the first LSTM state, because the LSTM is bidir and the first word is the most important for dialog act recognition
                rnndialinputs.append(uttout[:,0,:])

            rnnins = torch.cat([f.view(1,1,-1) for f in rnndialinputs],0)
            if self.useRNN: outputrnn, _ = self.dialrnn(rnnins)
            else: outputrnn = rnnins
            # got batch=1 X time=nutts X hsize

            outputmlp = [mlp(outputrnn).view(1,nutts,-1) for mlp in self.mlps]
            out1.append(outputmlp[0])
            if len(outputmlp)>1: out2.append(outputmlp[1])
        outres1 = torch.cat(out1,0)
        outres2 = None
        if len(outputmlp)>1: outres2 = torch.cat(out2,0)
        return outres1, outres2

    def eval(self,x0s,y0s,y1s=None):
        x,y1,y2=[],[],[]
        for dial in range(len(x0s)):
            
            if self.usecuda: v=Variable(torch.cuda.LongTensor([y0s[dial]]))
            else: v=Variable(torch.LongTensor([y0s[dial]]))
            y1.append(v)
            if y1s!=None:
                if self.usecuda: v=Variable(torch.cuda.LongTensor([y1s[dial]]))
                else: v=Variable(torch.LongTensor([y1s[dial]]))
                y2.append(v)
            dx=[]
            for tweet in range(len(x0s[dial])):
                if self.usecuda: v=Variable(torch.cuda.LongTensor(x0s[dial][tweet]))
                else: v=Variable(torch.LongTensor(x0s[dial][tweet]))
                dx.append(v)
            x.append(dx)
        samplesidx = list(range(len(x)))
        hatys, goldys = [],[]
        hatys2, goldys2 = [],[]
        for i in range(len(samplesidx)):
            self.train(False)
            probas1, probas2 = self.forward([x[samplesidx[i]]])

            pr = probas1.data.cpu().numpy()[0]
            haty = [np.argmax(onepr) for onepr in pr]
            goldy = y1[samplesidx[i]].view(-1).data.cpu().numpy()
            hatys.append(haty)
            goldys.append(goldy)

            if y1s!=None:
                pr = probas2.data.cpu().numpy()[0]
                haty = [np.argmax(onepr) for onepr in pr]
                goldy = y2[samplesidx[i]].view(-1).data.cpu().numpy()
                hatys2.append(haty)
                goldys2.append(goldy)

        flathaty = [item for sublist in hatys for item in sublist]
        flatgldy = [item for sublist in goldys for item in sublist]
        f1DA, f1SE = 0,0
        if len(flatgldy)>0:
            if max(flatgldy)>5: f1DA = evalDA(flathaty,flatgldy)
            else: f1SE = evalSE(flathaty,flatgldy)
        flathaty = [item for sublist in hatys2 for item in sublist]
        flatgldy = [item for sublist in goldys2 for item in sublist]
        if len(flatgldy)>0:
            if max(flatgldy)>5: f1DA = evalDA(flathaty,flatgldy)
            else: f1SE = evalSE(flathaty,flatgldy)

        return f1DA,f1SE
 
    def dettrain(self,x0s,y0s, y1s=None, xdev=None,ydev1=None,ydev2=None, xtest=None,ytest1=None,ytest2=None):
        import torch.optim
        x,y1,y2=[],[],[]
        for dial in range(len(x0s)):
            if self.usecuda: v=Variable(torch.cuda.LongTensor([y0s[dial]]))
            else: v=Variable(torch.LongTensor([y0s[dial]]))
            y1.append(v)
            if y1s!=None:
                if self.usecuda: v=Variable(torch.cuda.LongTensor([y1s[dial]]))
                else: v=Variable(torch.LongTensor([y1s[dial]]))
                y2.append(v)
            dx=[]
            for tweet in range(len(x0s[dial])):
                if self.usecuda: v=Variable(torch.cuda.LongTensor(x0s[dial][tweet]))
                else: v=Variable(torch.LongTensor(x0s[dial][tweet]))
                dx.append(v)
            x.append(dx)

        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        samplesidx = list(range(len(x)))
        total_loss1=0
        total_loss2=0
        for i in range(len(samplesidx)):
            goldy = y1[samplesidx[i]].view(-1)
            self.train(False)
            probas1, probas2 = self.forward([x[samplesidx[i]]])
            inloss = probas1.view(-1,self.nlabs[0])
            loss = criterion(input=inloss, target=goldy)
            total_loss1 += loss.data.cpu().numpy()
            if y1s!=None:
                goldy2 = y2[samplesidx[i]].view(-1)
                inloss2 = probas2.view(-1,self.nlabs[1])
                loss2 = criterion2(input=inloss2, target=goldy2)
                total_loss2 += loss2.data.cpu().numpy()
        if y1s!=None:
            printMostFrequent(ytest1,ytest2)
        print("epoch init loss %f %f" % (total_loss1, total_loss2))
        # we always print here, in order: F1DA F1SE
        if not xdev==None: print("eval init %f %f " % self.eval(xdev,ydev1,ydev2))
        if not xtest==None: print("test init %f %f " % self.eval(xtest,ytest1,ytest2))
        for epoch in range(self.nepochs):
            shuffle(samplesidx)
            total_loss1=0
            total_loss2=0
            for i in range(len(samplesidx)):
                optimizer.zero_grad()
                self.train(True)
                probas1, probas2 = self.forward([x[samplesidx[i]]])
                goldy = y1[samplesidx[i]].view(-1)
                inloss = probas1.view(-1,self.nlabs[0])
                loss = [criterion(input=inloss, target=goldy)]
                total_loss1 += loss[0].data.cpu().numpy()

                if self.limitSE>=0 and i>=self.limitSE:
                    # reduce the training size for task SE
                    gradseq = [loss[0].data.new(1).fill_(1),loss[0].data.new(1).fill_(0)]
                    goldy2 = y2[samplesidx[i]].view(-1)
                    inloss2 = probas2.view(-1,self.nlabs[1])
                    loss.append(criterion2(input=inloss2, target=goldy2))
                    total_loss2 += loss[1].data.cpu().numpy()
                elif self.limitDA>=0 and i>=self.limitDA:
                    gradseq = [loss[0].data.new(1).fill_(0),loss[0].data.new(1).fill_(1)]
                    # goldy2 = y2[samplesidx[i]].view(-1)
                    # inloss2 = probas2.view(-1,self.nlabs[1])
                    # loss.append(criterion2(input=inloss2, target=goldy2))
                    # total_loss2 += loss[1].data.cpu().numpy()
                else:
                    if y1s!=None:
                        # Two tasks are active
                        goldy2 = y2[samplesidx[i]].view(-1)
                        inloss2 = probas2.view(-1,self.nlabs[1])
                        loss.append(criterion2(input=inloss2, target=goldy2))
                        total_loss2 += loss[1].data.cpu().numpy()

                        # try MultiTask with single random gradient activation
                        activegrad = np.random.randint(0,2)
                        if activegrad==0:
                            gradseq = [loss[0].data.new(1).fill_(1),loss[0].data.new(1).fill_(0)]
                        else:
                            gradseq = [loss[0].data.new(1).fill_(0),loss[0].data.new(1).fill_(1)]
                    else:
                        # single task training
                        gradseq = [loss[0].data.new(1).fill_(1)]
                torch.autograd.backward(loss, gradseq)
                optimizer.step()

            print("epoch %d loss %f %f" % (epoch, total_loss1, total_loss2))
            # we always print here, in order: F1DA F1SE
            if not xdev==None: print("eval %d %f %f" % ((epoch,) + self.eval(xdev,ydev1,ydev2)))
            if not xtest==None: print("test %d %f %f" % ((epoch,) + self.eval(xtest,ytest1,ytest2)))
            sys.stdout.flush()
        print("THEEND")

