from mod import *
from data import *
import sys
import random

fold    = int(sys.argv[1])
embed   = int(sys.argv[2])
hsize   = int(sys.argv[3])
lr      = float(sys.argv[4])
nepok   = int(sys.argv[5])
trsize  = int(sys.argv[6]) # trainsize in nmuber of dialogs
parms = "fold %d lr %f embed %d hsize %d nepok %d trainsize %d" % (fold,lr,embed,hsize,nepok,trsize)
print(parms)

# UNCOMMENT one of these 3 "corpus" lines, depending on the experiment you want to do:

# mono-task with dialog act:
# corpus = MastoData("DA")

# mono-tasl with sentiment analysis:
# corpus = MastoData("SE")

# multi-task:
corpus = MastoData("JO")


# you can edit either "limitDA" or "limitSE" next if you want to limit the maximum number of dialogs for one of the task
# by default (-1), the whole training corpus is used.

model  = Model(corpus.nvoc, corpus.nlabs, embed=embed, hsize=hsize, lr=lr, nepok=nepok, useRNN=True, limitDA=-1, limitSE=-1)

model.usecuda=torch.cuda.is_available()
if model.usecuda: model.cuda()

dx,dy1,dy2 = corpus.getDev(fold=fold)
tx,ty1,ty2 = corpus.getTest()
x,y1,y2 = corpus.getTrain(fold=fold)
print("totaltrsize %d" % len(x))
print("totaltesize %d" % len(tx))

# The corpus is shrink next to $trsize
# (for both tasks simultaneously: this is used to range the X-axis in the paper's figures
# this is different from limitDA / limitSE, which only affects a single task)
tridx = range(len(x))
random.shuffle(tridx)
idx=tridx[0:trsize]
xx = [x[i] for i in idx]
yy1 = [y1[i] for i in idx]
if y2!=None: yy2 = [y2[i] for i in idx]
else: yy2=y2

# training is done here:
model.dettrain(xx,yy1,y1s=yy2,xdev=dx,ydev1=dy1,ydev2=dy2,xtest=tx,ytest1=ty1,ytest2=ty2)

