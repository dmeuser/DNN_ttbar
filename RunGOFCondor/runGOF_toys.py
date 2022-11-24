import argparse
import os
import subprocess as sp

def runCmds_GOF_toys(varName, year, version, nToys=100):
	iPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/datacards/"+varName+"/"
	oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+varName+"_files/Results_"+varName+"/"
	
	cmd = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=saturated --cminDefaultMinimizerStrategy 0 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --cminPreScan -t "+str(nToys)+" --toysFreq -s -1"
	#  ~cmd = "combine -M GoodnessOfFit "+iPath+"ttbar_"+varName+".root --algo=AD --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 50.1 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --verbose 3 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --cminPreScan -t "+str(nToys)+" -s -1"
	print(oPath, cmd)
	
	toysGOF = sp.check_output(cmd.split(), cwd=oPath)
	print(toysGOF)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
	parser.add_argument('--nToys', type=int, help="Number of toys generated pre node")
	parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018")
	parser.add_argument('--version', type=str, help="treeVersion, such as v07")
	args = parser.parse_args()
	
	if args.n and args.year and args.version:
		if args.nToys:
			runCmds_GOF_toys(args.n, args.year, args.version, nToys=args.nToys)
		else:
			runCmds_GOF_toys(args.n, args.year, args.version)
	else:
		print("\nNo variable name, version or year given\n")
