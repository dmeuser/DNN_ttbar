import argparse
import os
import subprocess as sp
import ROOT
import CombineHarvester.CombineTools.plotting as plot
import argparse
import json
import matplotlib.pyplot as plt
from GOF_plotPValue import *

def plotGOFtests(varName, varNameLeg, year, version):
    iPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/datacards/"+varName+"/"
    oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+varName+"_files/Results_"+varName+"/"
    
    # combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH125.root higgsCombineTest.GoodnessOfFit.mH125.123456.root -m 125.0 -o gof.json
    # plotGof.py gof.json --statistic saturated --mass 125.0 -o gof_plot --title-right="my label"
    
    seedArr = sp.check_output(["ls"], cwd=oPath)
    toySeeds = [el[37:-5] for el in seedArr.split("\n") if "higgsCombineTest.GoodnessOfFit.mH125" in el]
    toySeeds = [int(t) for t in toySeeds if t!=""]
    #  ~toySeeds = [toySeeds[3]]
    
    commandList = ["combineTool.py","-M","CollectGoodnessOfFit","-m","125.0","-o","gof_"+varName+".json","--input","higgsCombineTest.GoodnessOfFit.mH125.root"]
    toys = []
    for toySeed in toySeeds:
        #  ~print(toySeed)
        toyName = "higgsCombineTest.GoodnessOfFit.mH125."+str(toySeed)+".root"
        commandList.append(toyName)
        toys.append(oPath+toyName)
    
    collGOF = sp.check_output(commandList, cwd=oPath)
    print(collGOF)
    
    plotGOF(oPath+"higgsCombineTest.GoodnessOfFit.mH125.root", toys, oPath+"PValue_"+varName+"_test", txtTL="", txtTR=varNameLeg, varName=varName, year=year)



legends = {
    "METunc_Puppi" : "#sigma_{MET}^{Puppi}",
    "PuppiMET_X" : "p_{x}^{miss, old Puppi}",
    "PuppiMET_Y" : "p_{y}^{miss, old Puppi}",
    "PuppiMET_xy_X" : "p_{x}^{miss, Puppi}",
    "PuppiMET_xy_Y" : "p_{y}^{miss, Puppi}",
    "MET_X" : "p_{x}^{miss, old PF}",
    "MET_Y" : "p_{y}^{miss, old PF}",
    "MET_xy_X" : "p_{x}^{miss, PF}",
    "MET_xy_Y" : "p_{y}^{miss, PF}",
    "MET_xy_Y_emu" : "p_{y}^{miss, PF}",
    "MET_xy_Y_mumu" : "p_{y}^{miss, PF}",
    "MET_xy_Y_ee" : "p_{y}^{miss, PF}",
    "CaloMET_X" : "p_{x}^{miss, Calo}",
    "CaloMET_Y" : "p_{y}^{miss, Calo}",
    "CaloMET" : "p_{T}^{miss, Calo}",
    "vecsum_pT_allJet_X" : "p_{x}^{all j}",
    "vecsum_pT_allJet_Y" : "p_{y}^{all j}",
    "Jet1_pX" : "p_{x}^{j1}",
    "Jet1_pY" : "p_{y}^{j1}",
    "MHT" : "MHT",
    "mass_l1l2_allJet" : "m_{l1,l2,all j",
    "Jet2_pX" : "p_{x}^{j2}",
    "Jet2_pY" : "p_{y}^{j2}",
    "mjj" : "m_{jj}",
    "n_Interactions" : "n_{Interactions}",
    "MT2" : "MT2",
    "dPhiMETleadJet_Puppi" : "#Delta#phi(p_{T}^{miss, Puppi}, Jet 1)",
    "Lep2_pX" : "p_{x}^{l2}",
    "Lep2_pY" : "p_{y}^{l2}",
    "HT" : "HT",    
    "Lep1_pX" : "p_{x}^{l_{1}}",
    "Lep1_pY" : "p_{y}^{l_{1}}",
    "vecsum_pT_allJet" : "p_{T}^{all j}",
    "nJets" : "n_{jets}",
    "Jet1_E" : "E_{jet 1}",
    "DeepMET_reso_X" : "p_{x}^{\rm miss, DReso}",
    "DeepMET_reso_Y" : "p_{y}^{\rm miss, DReso}",
    "DeepMET_resp_X" : "p_{x}^{\rm miss, DResp}",
    "DeepMET_resp_Y" : "p_{y}^{\rm miss, DResp}"
}


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=str, help="Variable name of corresponding datacard")
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07")
    args = parser.parse_args()
    
    if "_VS_" in args.n:
        legName = legends[args.n.split("_VS_")[0]]+" vs "+legends[args.n.split("_VS_")[1]]
    else:
        legName = legends[args.n]
    
    if args.n and args.year and args.version:
            plotGOFtests(args.n, legName, args.year, args.version)
    else:
            print("\nNo proper variable name or year given\n")
    
    #  ~plotGOFtests(
    #  ~for n in ["PuppiMET_xy_X", "PuppiMET_xy_Y"]:
        #  ~plotGOFtests(n,legends[n])
