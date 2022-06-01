import matplotlib.pyplot as plt
import numpy as np
import json
import os.path
import argparse

def plot_P_Vals(year, version):
    #  ~iPath = "/net/data_cms1b/user/nattland/top_analysis/2018/v07/output_framework/datacards/"+varName+"/"
    #  ~oPath = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+varName+"_files/Results_"+varName+"/"
    if year=="2017":
	lumi=41.5
    elif year=="2016_preVFP":
	lumi=19.5
    elif year=="2016_postVFP":
	lumi=16.8
    else:
	lumi=59.8
    #  ~varList = np.array(["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "CaloMET", "MT2", "mjj", "nJets", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY"])
    varList = np.array(["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "mjj", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY"])
    
    niceList = np.array(["$"+legends[i]+"$" for i in varList])
    
	
    varList2D = []
    for i, el1 in enumerate(varList):
	    for j, el2 in enumerate(varList[i+1:]):
		    varList2D.append(el1+"_VS_"+el2)
    #  ~totList = varList + varList2D
    #  ~valArr = []

    pArr = []
    failedArr = []
    
    for var in np.concatenate((varList, varList2D)):
	fPath = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+var+"_files/Results_"+var+"/gof_"+var+".json"
	#  ~print(var)
	if os.path.exists(fPath):
	    with open(fPath) as f:
		data = json.load(f)
		#  ~print(data)
		pArr.append([var, data["125.0"]["p"]])
	else:
	    failedArr.append(var)

    
    print(failedArr)
    pMat = np.zeros((len(varList), len(varList)))-1
    xBinC = np.linspace(0.5, len(varList)-0.5, len(varList))
    yBinC = np.linspace(0.5, len(varList)-0.5, len(varList))
    
    for pVal in pArr:
	if "_VS_" in pVal[0]:
	    ind1 = np.argwhere(varList==pVal[0].split("_VS_")[0])[0][0]
	    ind2 = np.argwhere(varList==pVal[0].split("_VS_")[1])[0][0]
	    pMat[ind1, ind2] = pVal[1]*100
	    pMat[ind2, ind1] = pVal[1]*100
	else:
	    pMat[np.argwhere(varList==pVal[0])[0][0], np.argwhere(varList==pVal[0])[0][0]] = pVal[1]*100
    
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.hist(pMat.flatten(), bins=20, range=(0,100))
    ax.set_ylabel("number of GOF tests")
    ax.set_xlabel("p-Value in %")
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    ax.text(1.,1.,r"$59.7\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
    #  ~plt.tight_layout()
    plt.savefig("p_vals_histo"+year+".pdf")
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    mesh1 = ax.pcolormesh(range(len(varList)+1), range(len(varList)+1), pMat, vmin=0., vmax=100., cmap=plt.get_cmap("viridis"))
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("p-value (%)")
    for i,col in enumerate(pMat):
	    for j,vali in enumerate(col):
		    if vali<5:
			    ax.text(xBinC[j]-0.45, yBinC[i]-0.25, "{:.1f}".format(vali), fontsize=9, color="red",fontweight="bold")
		    #  ~else: 
			    #  ~ax.text(xBinC[j]-0.4, yBinC[i]-0.25, "-"+"{:.2f}".format(vali)[formI+1:], fontsize=12, color="green", fontweight="bold")
    
    plt.yticks(np.linspace(0.5, len(varList)-0.5, len(varList)), niceList, fontsize=9)
    plt.xticks(np.linspace(0.5, len(varList)-0.5, len(varList)), niceList, fontsize=9, rotation=90)
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"             Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    ax.text(1.,1.,str(lumi)+r"$\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
    plt.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.12)
    plt.savefig("p_vals_"+year+"_allUnc_short.pdf")



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
    "CaloMET_X" : "p_{x}^{miss, Calo}",
    "CaloMET_Y" : "p_{y}^{miss, Calo}",
    "CaloMET" : "p_{T}^{miss, Calo}",
    "vecsum_pT_allJet_X" : "p_{x}^{all jets}",
    "vecsum_pT_allJet_Y" : "p_{y}^{all jets}",
    "Jet1_pX" : "p_{x}^{Jet 1}",
    "Jet1_pY" : "p_{y}^{Jet 1}",
    "MHT" : "MHT",
    "mass_l1l2_allJet" : "m_{l1,l2,all jets}",
    "Jet2_pX" : "p_{x}^{Jet 2}",
    "Jet2_pY" : "p_{y}^{Jet 2}",
    "mjj" : "m_{jj}",
    "n_Interactions" : "n_{Interactions}",
    "MT2" : "MT2",
    "dPhiMETleadJet_Puppi" : "#Delta#phi(p_{T}^{miss, Puppi}, Jet 1)",
    "Lep2_pX" : "p_{x}^{l 2}",
    "Lep2_pY" : "p_{y}^{l 2}",
    "HT" : "HT",    
    "Lep1_pX" : "p_{x}^{l 1}",
    "Lep1_pY" : "p_{y}^{l 1}",
    "vecsum_pT_allJet" : "p_{T}^{all jets}",
    "nJets" : "n_{jets}",
    "Jet1_E" : "E_{jet 1}",
}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07")
    args = parser.parse_args()
    
    if args.year and (args.year in ["2016_preVFP", "2016_postVFP", "2017", "2018"]) and args.version:
	plot_P_Vals(args.year, args.version)
    else:
	print("No correct year/version given")
    
