import argparse
import os
import subprocess as sp

def runCmds_GOF_data(varName, year, version):
	iPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/datacards/"+varName+"/"
	oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+varName+"_files/Results_"+varName+"/"
	
	cmd_t2w = "text2workspace.py --channel-masks "+iPath+"ttbar_"+varName+".txt -o "+iPath+"ttbar_"+varName+".root"
	t2w = sp.check_output(cmd_t2w.split())
	print(t2w)
	
	#  ~cmd_dataGOF = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=saturated --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.01 --setParameters r=1 --setParameterRanges r=-10.,10. -m 125"
	#  ~cmd_dataGOF = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=saturated --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --cminPreFit 1 --cminPreScan -v 3"
	#  ~cmd_dataGOF = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=saturated --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 50.1 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --verbose 3 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --cminPreScan"
	
	cmd_dataGOF = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=saturated --cminDefaultMinimizerStrategy 0 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --cminPreScan"
	print(oPath)
	print(cmd_dataGOF)
	
	dataGOF = sp.check_output(cmd_dataGOF.split(), cwd=oPath)
	#Asimov: -t = -1
	print(dataGOF)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
	parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018")
	parser.add_argument('--version', type=str, help="treeVersion, such as v07")
	args = parser.parse_args()
	
	if args.n and args.year and args.version:
		runCmds_GOF_data(args.n, args.year, args.version)
	else:
		print("\nNo variable name or year given\n")

	
