import argparse
import os
import subprocess as sp
import ROOT
import CombineHarvester.CombineTools.plotting as plot
import argparse
import json
import matplotlib.pyplot as plt
from GOF_plotPValue import *

def plotImpacts_GOF(varName, year, version, makeJson=True, runAsimov=False):
    iPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/datacards/"+varName+"/"
    oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+varName+"_files/Results_"+varName+"/impacts/"
    
    if not os.path.exists(oPath):
        os.makedirs(oPath)
    
    asimovOpt = "-t -1" if runAsimov else ""
    asimovStr = "asimov_" if runAsimov else ""
    
    if makeJson:
	print("performing initial fit...")
	iniFit = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/output_framework/datacards/{name}/ttbar_{name}.root -m 125 --doInitialFit --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerPrecision 1E-12 --setParameters r=1 --setParameterRanges r=-2.,2. --X-rtd MINIMIZER_MaxCalls=999999999 --robustFit 1 {asimov}".format(name=varName, year=year, version=version, asimov=asimovOpt).split()),cwd=oPath)
	print(iniFit)
	doFits = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/output_framework/datacards/{name}/ttbar_{name}.root -m 125 --doFits --robustFit 1 --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerPrecision 1E-12 --X-rtd MINIMIZER_MaxCalls=999999999 --setParameters r=1 --setParameterRanges r=-2.,2. --parallel 7 {asimov}".format(name=varName, year=year, version=version, asimov=asimovOpt).split()),cwd=oPath)
	print(doFits)
	mJson = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/output_framework/datacards/{name}/ttbar_{name}.root -m 125 -o {path}Impacts_{asimov}{name}.json".format(name=varName, path=oPath, year=year, version=version, asimov=asimovStr).split()),cwd=oPath)
	print(mJson)
    print("plotImpacts.py -i {path}Impacts_{asimov}{name}.json -o impacts_{name}_{year}".format(name=varName, path=oPath, year=year, asimov="asimov_" if runAsimov else ""))
    doPlot = sp.check_output(("plotImpacts.py -i {path}Impacts_{asimov}{name}.json -o impacts_{asimov}{name}_{year}".format(name=varName, path=oPath, year=year, asimov=asimovStr).split()), cwd = oPath)
    print(doPlot)


    
if __name__ == "__main__":
    #  ~parser = argparse.ArgumentParser()
    #  ~parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
    #  ~args = parser.parse_args()
    
    #  ~if args.n and legends[args.n]!=None:
	    #  ~plotImpacts_GOF(args.n,legends[args.n])
    #  ~else:
	    #  ~print("\nNo proper variable name given\n")
    
    
    #  ~for n in ["PuppiMET_xy_X_VS_Lep1_pX"]:
    for n in ["DNN_MET_pT"]:
    #  ~for n in ["nJets"]:
        plotImpacts_GOF(n, "2018", "v08", makeJson=True, runAsimov=False)
        #  ~plotImpacts_GOF(n, "2018", "v08", makeJson=True, runAsimov=True)
        #  ~plotImpacts_GOF(n, "2017", "v05", makeJson=True, runAsimov=False)
        #  ~plotImpacts_GOF(n, "2016_preVFP", "v03", makeJson=True, runAsimov=False)
        #  ~plotImpacts_GOF(n, "2016_postVFP", "v03", makeJson=True, runAsimov=False)
