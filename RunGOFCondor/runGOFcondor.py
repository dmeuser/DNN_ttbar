import argparse
import os
import subprocess as sp

def runGOFtest(singleVar=None, makeBashs=False, dim=0, year="2018", version="v07"):
	
	clusterFileDir = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"
	# List of input features
	varList = ["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "CaloMET", "MT2", "mjj", "nJets", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY"]
	
	# Creating list of input feature combinations
	varList2D = []
	for i, el1 in enumerate(varList):
		for j, el2 in enumerate(varList[i+1:]):
			varList2D.append(el1+"_VS_"+el2)
	
	if dim==2:
		varList=varList2D
	elif dim!=1:
		varList+=varList2D
	
	# running for only one input feature, can also be a 2D combination
	if singleVar:
		varList = []
		varList.append(singleVar)
	
	
	for var in varList:
		pathList = [clusterFileDir+year+"/"+var+"_files",
					clusterFileDir+year+"/"+var+"_files/CombineOutLogErr",
					clusterFileDir+year+"/"+var+"_files/Results_"+var,
					clusterFileDir+year+"/"+var+"_files/CombineOutLogErr/runGOF_data",
					clusterFileDir+year+"/"+var+"_files/CombineOutLogErr/runGOF_toys",
					clusterFileDir+year+"/"+var+"_files/CombineOutLogErr/plotGOF_chi"]
		# Create necessary folders
		for filePath in pathList:
			if not os.path.exists(filePath):
				os.makedirs(filePath)
		
		# Create DAG file; runGOF_data_{}.sub is executed first, then 
		# runGOF_toys_{}.sub, then plotGOF_{}.sub
		with open(clusterFileDir+year+"/"+var+"_files/runGOF_"+var+".dag","w") as f:
			f.write("""
JOB D {pathName}/runGOF_data_{varName}.sub
JOB T {pathName}/runGOF_toys_{varName}.sub
JOB P {pathName}/plotGOF_chi_{varName}.sub

PARENT D CHILD T
PARENT T CHILD P
""".format(pathName=clusterFileDir+year+"/"+var+"_files",varName=var))
		
		# for each of the submissions, the sub and sh files are created
		for subName, subLength, subNr in zip(["runGOF_data", "runGOF_toys", "plotGOF_chi"], ["espresso", "espresso", "espresso"], [1, 5, 1]):
			with open(clusterFileDir+year+"/"+var+"_files/"+subName+"_"+var+".sub","w") as f:
				f.write("""
Universe                = vanilla
executable              = {path}{subName}.sh
arguments               = {varName} {year} {version}
output                  = {path}{year}/{varName}_files/CombineOutLogErr/{subName}/{subName}.$(ClusterId).$(ProcId).out
error                   = {path}{year}/{varName}_files/CombineOutLogErr/{subName}/{subName}.$(ClusterId).$(ProcId).err
log                     = {path}{year}/{varName}_files/CombineOutLogErr/{subName}/{subName}.$(ClusterId).log
stream_output           = True
request_CPUs 		= 1
+JobFlavour 		= {subLength}
queue {subNr}
""".format(path=clusterFileDir, year=year, varName=var, subName=subName, subLength=subLength, subNr=subNr, version=version))
	# new bash files are created, if necessary 
	if makeBashs:
		for subName in ["runGOF_data", "runGOF_toys", "plotGOF_chi"]:
			with open(clusterFileDir+subName+".sh","w") as f:
				f.write("""
#!/bin/bash
baseDir=/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/
cd $baseDir

export SCRAM_ARCH=slc7_amd64_gcc820
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/home4/institut_1b/nattland/CMSSW_10_5_0
eval `scramv1 runtime -sh`

cd $baseDir
python {subName}.py --n "$1" --y "$2" --v "$3"
	""".format(subName=subName, year=year, version=version))

	for var in varList:
		# old dag submission files are deleted
		for oldFileName in [".dag.condor.sub", ".dag.dagman.log", ".dag.dagman.out", ".dag.metrics", ".dag.lib.err", ".dag.nodes.log", ".dag.dagman.log", ".dag.lib.out", ".dag.rescue001"]:
			if os.path.isfile(clusterFileDir+year+"/"+var+"_files/runGOF_"+var+oldFileName):
				sp.call(["rm","runGOF_"+var+oldFileName],cwd=clusterFileDir+year+"/"+var+"_files")
		# dag files are submitted to condor
		condSub = sp.call(["condor_submit_dag",clusterFileDir+year+"/"+var+"_files/runGOF_"+var+".dag"])
		print(condSub)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--singleVar', type=str, help="Calculate GOF for only this variable")
	parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018")
	parser.add_argument('--version', type=str, help="treeVersion, such as v07")
	parser.add_argument('--dim', type=str, help="dimensions, if 1D use 1, if 2D use 2, else both")
	args = parser.parse_args()
	
	
	if args.year and (args.year in ["2016_preVFP", "2016_postVFP", "2017", "2018"]) and args.version:
		if args.singleVar:
			runGOFtest(singleVar=args.singleVar, year=args.year, version=args.version)
		elif args.dim:
			runGOFtest(dim=args.dim, year=args.year, version=args.version)
		else:
			runGOFtest(year=args.year, version=args.version)
	else: 
		print("No correct year name given")
