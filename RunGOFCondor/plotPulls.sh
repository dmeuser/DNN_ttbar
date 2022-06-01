#!/bin/bash
baseDir=/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/
cd $baseDir

export SCRAM_ARCH=slc7_amd64_gcc820
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/home4/institut_1b/nattland/CMSSW_10_5_0
eval `scramv1 runtime -sh`

text2workspace.py /net/data_cms1b/user/nattland/top_analysis/$2/$3/output_framework/datacards/$1/ttbar_$1.txt -m 125
combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/$2/$3/output_framework/datacards/$1/ttbar_$1.root -m 125 --doInitialFit --robustFit 1 --cminDefaultMinimizerStrategy 0  --setParameters r=1 --setParameterRanges r=-2.,2. --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 
combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/$2/$3/output_framework/datacards/$1/ttbar_$1.root -m 125 --doFits --robustFit 1 --cminDefaultMinimizerStrategy 0  --setParameters r=1 --setParameterRanges r=-2.,2. --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --parallel 6
combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/$2/$3/output_framework/datacards/$1/ttbar_$1.root -m 125 -o /home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/Impacts_$1_$2_noMC.json
