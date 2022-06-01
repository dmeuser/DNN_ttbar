import argparse
import os
import subprocess as sp
import ROOT
import CombineHarvester.CombineTools.plotting as plot
import argparse
import json
import matplotlib.pyplot as plt
from GOF_plotPValue import *

def plotImpacts_GOF(varName, makeJson=True):
    iPath = "/net/data_cms1b/user/nattland/top_analysis/2018/v07/output_framework/datacards/"+varName+"/"
    oPath = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/2018/"+varName+"_files/Results_"+varName+"/"
    
    if makeJson:
	print("preforming initial fit...")
	iniFit = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/2018/v06/output_framework/datacards/{name}/ttbar_{name}_1d_mumu.root -m 125 --doInitialFit --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.01 --setParameters r=1 --setParameterRanges r=-10.,10. --robustFit 1".format(name=varName).split()))
	print(iniFit)
	doFits = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/2018/v06/output_framework/datacards/{name}/ttbar_{name}_1d_mumu.root -m 125 --doFits --robustFit 1 --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.01 --setParameters r=1 --setParameterRanges r=-10.,10. --parallel 7".format(name=varName).split()))
	print(doFits)
	mJson = sp.check_output(("combineTool.py -M Impacts -d /net/data_cms1b/user/nattland/top_analysis/2018/v06/output_framework/datacards/{name}/ttbar_{name}_1d_mumu.root -m 125 -o {path}Impacts_{name}_mumu_only.json".format(name=varName, path=oPath).split()))
	print(mJson)
    print("plotImpacts.py -i {path}Impacts_{name}_mumu_only.json -o impacts_{name}_mumu_only".format(name=varName, path=oPath))
    #  ~doPlot = sp.check_output(("plotImpacts.py -i {path}Impacts_{name}_mumu_only.json -o impacts_{name}_mumu_only".format(name=varName, path=oPath).split()), cwd = oPath)
    #  ~print(doPlot)


    
if __name__ == "__main__":
    #  ~parser = argparse.ArgumentParser()
    #  ~parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
    #  ~args = parser.parse_args()
    
    #  ~if args.n and legends[args.n]!=None:
	    #  ~plotImpacts_GOF(args.n,legends[args.n])
    #  ~else:
	    #  ~print("\nNo proper variable name given\n")
    
    
    for n in ["MET_xy_Y"]:
        plotImpacts_GOF(n, makeJson=False)
