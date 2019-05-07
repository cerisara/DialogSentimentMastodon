#!/bin/bash 

# input = datatrainJoint.idx datatrainJoint.txt 
# output = tmptrain.tootids.* tmpdev.tootids.* tmptrain.wds.* tmpdev.wds.*
for i in 0 1 2 3 4 5 6 7 8 9; do
    python splitTrainDev.py $i
done
exit
