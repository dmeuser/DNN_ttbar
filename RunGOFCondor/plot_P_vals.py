import matplotlib.pyplot as plt
import numpy as np
import json
import os.path
import argparse

def plot_P_Vals(year, version):
    if year=="2017":
        lumi=41.5
    elif year=="2016_preVFP":
        lumi=19.5
    elif year=="2016_postVFP":
        lumi=16.8
    else:
        lumi=59.8
    #  ~varList = np.array(["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "CaloMET", "MT2", "mjj", "nJets", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY"])
    #  ~varList = np.array(["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "CaloMET", "MT2", "mjj", "nJets", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY", "DeepMET_reso_X", "DeepMET_reso_Y", "DeepMET_resp_X", "DeepMET_resp_Y"])
    varList = np.array(["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "mjj", "Jet1_E", "HT", "Jet2_pX", "Jet2_pY"])
    
    niceList = np.array([legends[i] for i in varList])
    
        
    varList2D = []
    for i, el1 in enumerate(varList):
            for j, el2 in enumerate(varList[i+1:]):
                    varList2D.append(el1+"_VS_"+el2)
    #  ~totList = varList + varList2D
    #  ~valArr = []

    pArr = []
    failedArr = []
    
    for var in np.concatenate((varList, varList2D)):
        fPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+var+"_files/Results_"+var+"/gof_"+var+".json"
        #  ~fPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/GOFtestResults//"+var+"_files/Results_"+var+"/gof_"+var+".json"
	
        if os.path.exists(fPath):
            with open(fPath) as f:
                try:
                    data = json.load(f)
                except ValueError:
                    print(var," has no valid output")
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
    
    #  ~fig, ax = plt.subplots(1,1, figsize=(5,4))
    #  ~ax.hist(pMat.flatten(), bins=20, range=(0,100))
    #  ~ax.set_ylabel("number of GOF tests")
    #  ~ax.set_xlabel("p-Value in %")
    #  ~ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    #  ~ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    #  ~ax.text(1.,1.,r"$59.7\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
    #plt.tight_layout()
    #  ~plt.savefig("p_vals_histo"+year+".pdf")
    
    #  ~fig, ax = plt.subplots(1,1, figsize=(5,4.5))
    fig, ax = plt.subplots(1,1, figsize=(5.2,4.7))
    mesh1 = ax.pcolormesh(range(len(varList)+1), range(len(varList)+1), pMat, vmin=0., vmax=100., cmap=plt.get_cmap("viridis"))
    cbar = fig.colorbar(mesh1, ax=ax, pad=0.02)
    cbar.set_label("p-value (%)")
    for i,col in enumerate(pMat):
            for j,vali in enumerate(col):
                    if vali<5:
                            ax.text(xBinC[j]-0.46, yBinC[i]-0.2, "{:.1f}".format(vali), fontsize=6.5, color="red",fontweight="bold")
                    #  ~else: 
                            #  ~ax.text(xBinC[j]-0.4, yBinC[i]-0.25, "-"+"{:.2f}".format(vali)[formI+1:], fontsize=12, color="green", fontweight="bold")
    
    plt.yticks(np.linspace(0.5, len(varList)-0.5, len(varList)), niceList, fontsize=9)
    plt.xticks(np.linspace(0.5, len(varList)-0.5, len(varList)), niceList, fontsize=9, rotation=90)
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.002,r"            Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=11, color=(0.3,0.3,0.3))
    ax.text(1.,1.002,str(lumi)+r"$\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
    plt.subplots_adjust(left=0.14, right=0.978, top=0.955, bottom=0.155)
    #  ~plt.tight_layout(pad=0.1)
    plt.savefig("p_val_plots/final_plots/p_vals_"+year+"_lessUnc_final.pdf")
    #  ~plt.savefig("p_val_plots/final_plots/p_vals_"+year+"_allUnc_final.pdf")



legends = {
    "METunc_Puppi" : r"$\sigma_{MET}^{Puppi}$",
    "PuppiMET_X" : r"$p_{x}^{miss, old Puppi}$",
    "PuppiMET_Y" : r"$p_{y}^{miss, old Puppi}$",
    "PuppiMET_xy_X" : r"$p_{x}^{\rm miss, Puppi}$",
    "PuppiMET_xy_Y" : r"$p_{y}^{\rm miss, Puppi}$",
    "MET_X" : r"$p_{x}^{\rm miss, old PF}$",
    "MET_Y" : r"$p_{y}^{\rm miss, old PF}$",
    "MET_xy_X" : r"$p_{x}^{\rm miss, PF}$",
    "MET_xy_Y" : r"$p_{y}^{\rm miss, PF}$",
    "CaloMET_X" : r"$p_{x}^{\rm miss, Calo}$",
    "CaloMET_Y" : r"$p_{y}^{\rm miss, Calo}$",
    "CaloMET" : r"$p_{\rm T}^{\rm miss, Calo}$",
    "vecsum_pT_allJet_X" : r"$p_{x}^{all\,\,j}$",
    "vecsum_pT_allJet_Y" : r"$p_{y}^{all\,\,j}$",
    "Jet1_pX" : r"$p_{x}^{\,j_1}$",
    "Jet1_pY" : r"$p_{y}^{\,j_1}$",
    "MHT" : r"$m_{\rm HT}$",
    "mass_l1l2_allJet" : r"$m_{l1,l2,all\,j}$",
    "Jet2_pX" : r"$p_{x}^{\,j_2}$",
    "Jet2_pY" : r"$p_{y}^{\,j_2}$",
    "mjj" : r"$m_{jj}$",
    "n_Interactions" : r"$n_{Interactions}$",
    "MT2" : r"$M_{\rm T2}$",
    "dPhiMETleadJet_Puppi" : r"$\Delta\phi(p_{T}^{miss, Puppi}, j_1)$$",
    "Lep2_pX" : r"$p_{x}^{\,l_2}$",
    "Lep2_pY" : r"$p_{y}^{\,l_2}$",
    "HT" : r"$H_{\rm T}$",    
    "Lep1_pX" : r"$p_{x}^{\,l_1}$",
    "Lep1_pY" : r"$p_{y}^{\,l_1}$",
    "vecsum_pT_allJet" : r"$p_{T}^{all\,j}$",
    "nJets" : r"$n_{jets}$",
    "Jet1_E" : r"$E_{\,j_{1}}$",
    "DeepMET_reso_X" : r"$p_{x}^{\rm miss, DReso}$",
    "DeepMET_reso_Y" : r"$p_{y}^{\rm miss, DReso}$",
    "DeepMET_resp_X" : r"$p_{x}^{\rm miss, DResp}$",
    "DeepMET_resp_Y" : r"$p_{y}^{\rm miss, DResp}$"
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
    
