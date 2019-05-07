# This is the main script to run experiments with the hierarchical RNN on dialog act recognition and sentiment classification of Mastodon Twitter-like social network dialogs.
# By default, it will run a unique experiment over a single fold with a joint dialog-act & sentiment model.

XPtype="fast"

# If you want to reproduce the experiments of the paper
# XPtype="crossvalidation"

# requirements:
# - linux OS (tested on Ubuntu 16.04.4 LTS) with bash and python 2.7.12
# - numpy,pytorch (tested with version 0.3.0.post4) must be installed within a virtualenv located at $HOME/envs/pytorch

# edit the following variable to the location of your virtual environment with pytorch, if pytorch is not installed globally
envdir="$HOME/envs/pytorch/bin/activate"

if [ ! -f "$envdir" ]; then
    echo "ERROR: you must have pytorch installed in a virtualenv in $envdir"
    exit
fi

rm -f logdase.*

curdir=""$(pwd)
echo "#!/bin/bash" > runda.sh
echo 'source "'$envdir'"' >> runda.sh
echo 'cd "'$curdir'"' >> runda.sh
echo 'i=$(echo $1 | cut -d_ -f1)' >> runda.sh
echo 'j=$(echo $1 | cut -d_ -f2)' >> runda.sh
echo 'echo "fold $i trainsize $j"' >> runda.sh
# run over 500 epochs
echo 'python xpMT.py $i 100 100 0.001 500 $j | tee logdase.$i.$j' >> runda.sh
chmod 755 runda.sh

if [ "$XPtype" == "fast" ]; then
    # run a single experiment on Fold 0 with training corpus of size 1 (see Figure 2, part of the first point on the left)
    bash ./runda.sh "0_1"
else
    # run 10 fold-crossvalidation with varying training corpus size (see Figure 2 of the paper)
    # launch ./runda.sh 140 times with various parameters
    # the results will be saved in logdase.fold.tsize
    for i in 0 1 2 3 4 5 6 7 8 9; do
       for tsize in 1 10 50 100 150 200 100000; do
           parms=$i"_"$tsize
           bash ./runda.sh $parms
           # The same experiment is run a second time to avoid badly initialized parameters
           parms=$i"_0"$tsize
           bash ./runda.sh $parms
       done
    done
fi

# compute the average accuracy for dialog act recognition and sentiment analysis, when the target task (optimized on the dev set) is dialog act recognition
python analyzeda.py ./ 0 | tail -1

# compute the average accuracy for dialog act recognition and sentiment analysis, when the target task (optimized on the dev set) is sentiment analysis
python analyzeda.py ./ 1 | tail -1

