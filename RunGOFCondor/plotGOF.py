import argparse
import os
import subprocess as sp

def plotGOFtests(varName):
	iPath = "/net/data_cms1b/user/nattland/top_analysis/2018/v06/output_framework/datacards/"+varName+"/"
	oPath = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+varName+"_files/Results_"+varName+"/"
	
	# combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH125.root higgsCombineTest.GoodnessOfFit.mH125.123456.root -m 125.0 -o gof.json
	# plotGof.py gof.json --statistic saturated --mass 125.0 -o gof_plot --title-right="my label"
	
	seedArr = sp.check_output(["ls"], cwd=oPath)
	toySeeds = [el[37:-5] for el in seedArr.split("\n") if "higgsCombineTest.GoodnessOfFit.mH125" in el]
	toySeeds = [int(t) for t in toySeeds if t!=""]
	
	commandList = ["combineTool.py","-M","CollectGoodnessOfFit","-m","125.0","-o","gof_"+varName+".json","--input","higgsCombineTest.GoodnessOfFit.mH125.root"]
	for toySeed in toySeeds:
		print(toySeed)
		commandList.append("higgsCombineTest.GoodnessOfFit.mH125."+str(toySeed)+".root")
	
	collGOF = sp.check_output(commandList, cwd=oPath)
	print(collGOF)
	
	plotGOF = sp.check_output(["plotGof.py","gof_"+varName+".json","--statistic","saturated","--mass","125.0","-o","gof"+varName+"_plot","--title-right="+varName], cwd=oPath)
	print(plotGOF)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
	args = parser.parse_args()
	
	if args.n:
		plotGOFtests(args.n)
	else:
		print("\nNo variable name given\n")
