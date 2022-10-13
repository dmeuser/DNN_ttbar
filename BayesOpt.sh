#!/bin/bash
baseDir=/afs/cern.ch/user/p/phnattla/DNN_ttbar
cd $baseDir
pwd
source setup_tf241.sh
python BayesOpt.py --year $1 --version $2 --probe """$3"""
